"""
brain_service - the first behaviour loop.

Spec section 13: perceive, update world state, notice what matters, decide
whether to act, act, remember. This is the smallest honest version of that
loop, and it does exactly one social thing - it notices a stranger, asks who
they are, and remembers the answer:

    UNKNOWN_PERSON_DETECTED
        -> "Hello. I don't think we've met. What's your name?"
    HEARD "Ich heiße Pierre"
        -> vision_service enrols the views it has already gathered
        -> PERSON_NAMED
        -> "Nice to meet you, Pierre."

Next time that face appears the gallery recognises it, KNOWN_PERSON_DETECTED
arrives instead, and aiRon greets Pierre by name without asking anything.

The rules here are hard-coded, which is the point rather than a shortcut. An
LLM will make these decisions in Phase 4, but the loop it plugs into wants to
exist first, and wants to be the thing that owns cooldowns, turn-taking and
what the face is doing - none of which should ever depend on a model
responding in time (spec section 20).

aiRon says the name back before it is worth anything, deliberately. Neither
the speech recogniser nor the name parser can tell a name from any other noun,
so the only real check is the human hearing "Nice to meet you, Müde" and
saying no. Enrolment happens first because the face views are already banked
and expire when the person leaves; a wrong name is cheap to delete and a
missed enrolment is not.
"""

from __future__ import annotations

import random
import threading
import time

from ..core import EventBus, EventType, log
from ..memory.service import LONG_ABSENCE_S
from .naming import is_refusal, parse_name
from .tools import Toolbox

#: Do not greet the same person again for this long. A real brain will decide
#: this from "have I seen you today" once memory_service exists.
GREET_COOLDOWN_S = 120.0

#: How long to wait for an answer before letting it go.
#:
#: Refreshed on every sign that the exchange is still alive: aiRon finishing a
#: sentence, and the microphone hearing a voice start or stop. Without the
#: voice half, a long answer times out while it is still being given - the
#: detector only releases an utterance after a pause, and a transcript of a
#: seven-second segment arrived two seconds after aiRon had already said
#: "never mind" and walked away from the conversation.
ANSWER_TIMEOUT_S = 12.0

#: However lively the room, one exchange cannot run longer than this. Every
#: voice the array hears refreshes the answer deadline, so in a room with a
#: television on it would otherwise never come round - aiRon would sit in a
#: conversation nobody is having.
CONVERSATION_LIMIT_S = 45.0

#: Ask, and if that answer made no sense, ask once more. Never a third time -
#: a robot that keeps asking your name is not charming, it is broken.
MAX_ASKS = 2

#: Having asked a stranger once, leave them alone for a while.
ASK_COOLDOWN_S = 300.0

LINES = {
    "en": {
        "greet_known": ["Hello {name}, nice to see you.",
                        "Hi {name}, good to see you again."],
        "greet_again": ["Hello again, {name}.", "Good to see you back, {name}."],
        "greet_after_absence": ["{name}! It has been a while.",
                                "Hello {name}, I have not seen you in a while."],
        "greet_unknown": ["Hello there.", "Hi, nice to see you."],
        "ask_name": ["Hello. I don't think we've met. What's your name?",
                     "Hi. I don't know you yet. What should I call you?"],
        "ask_again": ["Sorry, I didn't catch that. What's your name?",
                      "I missed that. What should I call you?"],
        "named": ["Nice to meet you, {name}.", "Hello {name}. I'll remember you."],
        "declined": ["No problem.", "That's alright."],
        "gave_up": ["Never mind.", "Let's leave it for now."],
        "failed": ["Sorry, I couldn't get a good look at you.",
                   "I didn't see you well enough to remember that."],
    },
    "de": {
        "greet_known": ["Hallo {name}, schön dich zu sehen.",
                        "Hallo {name}, schön dass du da bist."],
        "greet_again": ["Hallo {name}, schön dich wiederzusehen.",
                        "Da bist du ja wieder, {name}."],
        "greet_after_absence": ["{name}! Lange nicht gesehen.",
                                "Hallo {name}, wir haben uns lange nicht gesehen."],
        "greet_unknown": ["Hallo!", "Schön dich zu sehen."],
        "ask_name": ["Hallo. Ich glaube wir kennen uns noch nicht. Wie heißt du?",
                     "Hallo. Dich kenne ich noch nicht. Wie soll ich dich nennen?"],
        "ask_again": ["Entschuldigung, das habe ich nicht verstanden. Wie heißt du?",
                      "Das habe ich nicht mitbekommen. Wie soll ich dich nennen?"],
        "named": ["Freut mich, {name}.", "Hallo {name}. Ich merke mir dich."],
        "declined": ["Kein Problem.", "Alles gut."],
        "gave_up": ["Auch gut.", "Lassen wir das."],
        "failed": ["Ich habe dich leider nicht gut genug gesehen.",
                   "Entschuldige, das konnte ich mir nicht merken."],
    },
}


#: How many fragments are folded into one turn. Enough for a sentence the VAD
#: chopped up; not so many that aiRon answers a paragraph nobody remembers
#: saying.
MAX_FRAGMENTS = 3

#: Quiet needed before aiRon accepts that a turn is over: nobody speaking, no
#: audio waiting to be transcribed, no decode running, and this long since the
#: last words arrived.
#:
#: Short, because it is only the tail of the wait. The expensive part is
#: waiting for the evidence to settle, and that ends by itself the moment the
#: room goes quiet and the last decode lands.
SETTLE_S = 0.4

#: How often the talker looks at whether the turn has finished.
POLL_S = 0.1


class BrainService:
    """
    Decides what aiRon does about the people it can see.

    Owns a small thread of its own, purely so a conversation can time out when
    nobody says anything. Everything else is driven by events.
    """

    def __init__(self, bus: EventBus, *, speech=None, vision=None, face=None,
                 memory=None, conversation=None, lang: str = "en",
                 can_listen: bool = False, ears=None, listener=None, store=None,
                 weather=None):
        self.bus = bus
        self.speech = speech
        self.vision = vision
        self.face = face
        self.memory = memory
        self.conversation = conversation
        # Not used to hear with - the bus already delivers that - but to ask
        # whether hearing has finished. See _turn_over.
        self.ears = ears
        self.listener = listener
        self.lang = lang if lang in LINES else "en"
        self.can_listen = can_listen
        # What the model may ask for. Constructed with only the four things it
        # is allowed to touch - not with self - so that the set of reachable
        # actions is visible here rather than implied by what a tool happens
        # to import (AIRON-8). After self.lang, which it needs.
        self.toolbox = Toolbox(memory=memory, face=face, store=store,
                               weather=weather, lang=self.lang)

        self._lock = threading.Lock()
        self._greeted: dict[str, float] = {}
        self._asked: dict[str, float] = {}

        self._asking: str | None = None
        self._asks = 0
        self._deadline = 0.0
        self._started = 0.0
        self._mirror_was: bool | None = None

        # Who is actually in front of the camera. aiRon only talks to somebody
        # who is there - the microphone hears the whole room, and a robot that
        # answers the television is worse company than one that says nothing.
        self._present: str | None = None

        # What has been heard but not yet answered. The model call runs on the
        # talker thread rather than the one that delivered the transcript, so
        # hearing carries on while aiRon is thinking about what to say.
        self._buffer: list[str] = []
        self._heard_at = 0.0

        self._thread = threading.Thread(target=self._run, name="brain", daemon=True)
        self._talker = threading.Thread(target=self._talk, name="brain-llm", daemon=True)
        self._stop = threading.Event()
        bus.subscribe(self._on_event)

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._thread.start()
        if self.conversation is not None:
            self._talker.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        if self._talker.is_alive():
            self._talker.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.wait(0.2):
            with self._lock:
                now = time.monotonic()
                expired = self._asking is not None and (
                    now > self._deadline or now > self._started + CONVERSATION_LIMIT_S)
            if expired:
                self._say("gave_up")
                self._end_conversation()

    # --------------------------------------------------------------- events

    def _on_event(self, event) -> None:
        kind = event.type
        if kind is EventType.PERSON_ENTERED:
            self._present = event.payload.get("person")
        elif kind is EventType.KNOWN_PERSON_DETECTED:
            self._present = event.payload.get("person")
            self._recognised_mid_question(event.payload.get("was"))
            self._greet(event.payload.get("person"), event.payload.get("name"),
                        was=event.payload.get("was"))
        elif kind is EventType.UNKNOWN_PERSON_DETECTED:
            self._met_a_stranger(event.payload.get("person"))
        elif kind is EventType.PERSON_NAMED:
            self._present = event.payload.get("person")
            # Already welcomed by name in _answer(); just make sure the
            # greeting logic does not say hello all over again.
            self._mark_greeted(event.payload.get("person"), event.payload.get("was"))
        elif kind is EventType.HEARD:
            self._heard(event.payload.get("text", ""))
        elif kind is EventType.PERSON_LEFT:
            self._person_left(event.payload.get("person"))
        elif kind is EventType.SPEECH_FINISHED:
            self._refresh_deadline()
        elif kind is EventType.VOICE_STARTED:
            self._refresh_deadline()
            self._look("listening")
        elif kind is EventType.VOICE_STOPPED:
            # Somebody just finished saying something; the transcript is a
            # second or so behind. Hold the conversation open for it.
            self._refresh_deadline()

    # ------------------------------------------------------------ behaviour

    def _greet(self, person: str | None, name: str | None, was: str | None = None) -> None:
        if person is None or not self._due(self._greeted, person, GREET_COOLDOWN_S):
            return
        # Someone greeted as a stranger and only then recognised has already
        # been said hello to; learning their name is not a second arrival.
        if was is not None and not self._due(self._greeted, was, GREET_COOLDOWN_S):
            self._mark_greeted(person, was)
            return
        self._mark_greeted(person)
        self._look("happy" if name else "curious")
        self._say(self._greeting_for(person, name), name=name)

    def _greeting_for(self, person: str, name: str | None) -> str:
        """
        Say hello like somebody who remembers you.

        Greeting a regular exactly as you greet a stranger is the tell that a
        robot is not really keeping track. memory_service already counts the
        visits and measures the gap; this only has to use them.
        """
        if name is None:
            return "greet_unknown"
        if self.memory is None:
            return "greet_known"
        visits, gap = self.memory.acquaintance(person)
        if visits <= 1:
            return "greet_known"
        return "greet_after_absence" if gap >= LONG_ABSENCE_S else "greet_again"

    def _met_a_stranger(self, person: str | None) -> None:
        """Ask who this is, if aiRon can actually hear and remember an answer."""
        if person is None:
            return
        if not (self.can_listen and self.speech is not None and self.vision is not None):
            self._greet(person, None)
            return
        with self._lock:
            if self._asking is not None:
                return                       # already mid-conversation
        if not self._due(self._asked, person, ASK_COOLDOWN_S):
            return

        with self._lock:
            self._asking = person
            self._asks = 1
            self._started = time.monotonic()
            self._asked[person] = self._started
            self._deadline = self._started + ANSWER_TIMEOUT_S
        self._suspend_mirror()
        self._look("curious")
        self._say("ask_name")

    def _answer(self, text: str) -> None:
        with self._lock:
            person = self._asking
        if person is None:
            return

        if is_refusal(text):
            self._say("declined")
            self._end_conversation()
            return

        name = parse_name(text)
        if name is None:
            with self._lock:
                self._asks += 1
                again = self._asks <= MAX_ASKS
                self._deadline = time.monotonic() + ANSWER_TIMEOUT_S
            if again:
                self._look("confused")
                self._say("ask_again")
            else:
                self._say("gave_up")
                self._end_conversation()
            return

        identity = self.vision.enrol_current(name)
        if identity is None:
            self._say("failed")
            self._end_conversation()
            return
        self._look("happy")
        self._say("named", name=identity.name)
        self._end_conversation()

    def _recognised_mid_question(self, was: str | None) -> None:
        """
        Recognition can land after aiRon has already started asking who this is.

        Drop the question silently. Saying "never mind" here would follow a
        greeting by name with an apology for a conversation aiRon started and
        then answered for itself, which is how the very first live run of this
        sounded - it asked André his name, worked it out, welcomed him, and
        then told him to forget it.
        """
        if was is None:
            return
        with self._lock:
            asking = self._asking
        if asking == was:
            self._end_conversation()

    def _person_left(self, person: str | None) -> None:
        if person == self._present:
            self._present = None
            self._restore_mirror()
        with self._lock:
            asking = self._asking
        if asking is not None and person == asking:
            self._end_conversation()

    # ----------------------------------------------------------- conversing

    def _heard(self, text: str) -> None:
        """Somebody said something. Either it answers the question aiRon asked,
        or it is conversation."""
        with self._lock:
            asking = self._asking
        if asking is not None:
            self._answer(text)
        else:
            self._consider(text)

    def _consider(self, text: str) -> None:
        """Add to what aiRon is being told. Not the same as answering it."""
        if self.conversation is None or self._present is None:
            return
        with self._lock:
            self._buffer.append(text)
            del self._buffer[:-MAX_FRAGMENTS]
            self._heard_at = time.monotonic()
        self._suspend_mirror()
        self._look("thinking")

    def _still_talking(self) -> bool:
        """Is there more of this turn still on its way?

        Three things say yes, and all three have been seen mid-sentence:
        the microphone can hear a voice right now; there is captured audio
        queued that nobody has transcribed; a decode is running.

        That last one is why a timer alone cannot do this. A decode takes
        one to three seconds, so the second half of a sentence reaches the
        brain long after the person finished saying it - and any wait short
        enough to feel conversational would expire before it arrived.
        """
        if self.ears is not None:
            if self.ears.hearing_voice or not self.ears.utterances.empty():
                return True
        return bool(self.listener is not None and self.listener.decoding)

    def _turn_over(self) -> str | None:
        """The whole turn as one line, once nobody is still saying it."""
        with self._lock:
            if not self._buffer:
                return None
            quiet_for = time.monotonic() - self._heard_at
        if self._still_talking() or quiet_for < SETTLE_S:
            return None
        with self._lock:
            text = " ".join(self._buffer)
            self._buffer.clear()
        return text

    def _talk(self) -> None:
        """
        One turn, one answer.

        The first version of this answered each utterance as it arrived and
        folded anything heard mid-thought into a follow-up. That produced
        exactly the thing it was meant to stop: "Ich hab dich gebaut" and "bin
        sozusagen dein Schöpfer" are one thought, and aiRon thanked him for it
        twice. The fix is not to answer sooner and merge afterwards - it is to
        wait until the person has actually finished.
        """
        while not self._stop.wait(POLL_S):
            text = self._turn_over()
            if text is None:
                continue
            try:
                self._reply_to(text)
            except Exception as exc:
                log(f"[brain] reply failed: {str(exc)[:120]}")

    def _reply_to(self, heard: str) -> None:
        person = self._present
        context = self.memory.context_for(person) if self.memory else ""
        history = self.memory.conversation() if self.memory else []

        said: list[str] = []
        wore: list[str] = []

        def wear(emotion: str) -> None:
            """The face, as soon as the model has chosen one - which is before
            it has finished choosing the words."""
            wore.append(emotion)
            self._look(emotion)

        def speak(sentence: str) -> bool:
            """One finished sentence, while the rest is still being written.

            Composed for whoever was there when they spoke. Seconds later that
            may not be who is standing here now - and the reply is built around
            their name and what aiRon remembers about them. Said to the wrong
            person it is not merely stale, it is wrong about who they are:
            André was told "Ja, Max, so heißt du doch" because Max had asked
            the question and left while the answer was being written.

            So it is abandoned rather than redirected. It cannot be rewritten
            for somebody else without asking the model again, and saying
            nothing to a person aiRon has not answered is a much smaller
            failure than calling them by another person's name. Checking here,
            per sentence, is what streaming buys: the turn can be cut off the
            moment the room changes instead of being judged once at the end.
            """
            if self._present != person:
                log(f"[brain] stopped mid-reply - it was for "
                    f"{person or 'nobody'}, and {self._present or 'nobody'} is here now")
                return False
            if self.speech is not None:
                self.speech.say(sentence, lang=self.lang)
            said.append(sentence)
            return True

        reply = self.conversation.reply(heard, context=context, history=history,
                                        on_emotion=wear, on_sentence=speak)
        if reply is None:
            # Nothing to say beats saying something wrong. The face drops the
            # thinking look so it does not sit there pretending.
            self._look("curious")
            self._restore_mirror()
            return

        # After the words, deliberately. look_at moves the eyes and the person
        # should see that while aiRon is still speaking, not instead of being
        # answered - and a tool that fails must not cost them the reply.
        if reply.actions:
            for outcome in self.toolbox.run(reply.actions, person=person):
                # A tool that fetched a fact says it here, in aiRon's voice
                # but not in the model's words. The weather is the only one
                # that does, and it does so precisely because the alternative
                # is asking a language model what tomorrow will be like.
                if outcome.speak and self.speech is not None and self._present == person:
                    self.speech.say(outcome.speak, lang=self.lang)

        if not wore:                     # the stream never got as far as a face
            self._look(reply.emotion)
        if not said:
            self._look("curious")
            self._restore_mirror()
        else:
            log(f"[brain] first words in {reply.first_words:.1f}s, "
                f"whole reply in {reply.seconds:.1f}s")

        self._remember(reply, person)

    def _remember(self, reply, person: str | None) -> None:
        """
        Keep what the model judged worth keeping.

        Filed against whoever actually said it, not whoever is in front of the
        camera now, and kept even when the reply itself went unsaid: they still
        told aiRon their favourite colour, whether or not it ever answered.
        """
        # The model decides what mattered; memory_service decides how long it
        # lasts. Neither could do the other's half.
        if self.memory is None or person is None:
            return
        for item in reply.remember:
            self.memory.remember(
                item["text"], person_id=person,
                kind=item.get("kind", "episodic"),
                importance=float(item.get("importance", 0.4)))

    # ------------------------------------------------------------- plumbing

    def _due(self, book: dict[str, float], key: str, cooldown: float) -> bool:
        with self._lock:
            return time.monotonic() - book.get(key, -1e9) >= cooldown

    def _mark_greeted(self, person: str | None, was: str | None = None) -> None:
        now = time.monotonic()
        with self._lock:
            if person is not None:
                self._greeted[person] = now
            if was is not None:
                self._greeted[was] = now

    def _refresh_deadline(self) -> None:
        """Anything that shows the exchange is still live buys it more time."""
        with self._lock:
            if self._asking is not None:
                self._deadline = time.monotonic() + ANSWER_TIMEOUT_S

    def _end_conversation(self) -> None:
        with self._lock:
            self._asking = None
            self._asks = 0
        self._restore_mirror()

    def _say(self, line: str, name: str | None = None) -> None:
        if self.speech is None:
            return
        text = random.choice(LINES[self.lang][line])
        self.speech.say(text.format(name=name) if name else text, lang=self.lang)

    def _look(self, emotion: str) -> None:
        if self.face is not None:
            self.face.command.emotion = emotion

    def _suspend_mirror(self) -> None:
        """
        Stop copying the human for the duration of a conversation.

        Mirroring is charming when aiRon is idle and wrong when it is talking
        to you: a face that pulls your expression back at you while asking a
        question reads as mockery rather than attention.
        """
        if self.face is None or self._mirror_was is not None:
            return
        self._mirror_was = self.face.mirror
        self.face.mirror = False

    def _restore_mirror(self) -> None:
        if self.face is not None and self._mirror_was is not None:
            self.face.mirror = self._mirror_was
        self._mirror_was = None
