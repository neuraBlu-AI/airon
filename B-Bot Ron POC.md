Angelehnt an den Film "Ron Gone Wrong" wollen wir einen Social Bot bauen, der mit im Haushalt lebt, neugierig ist, Fragen stellt und beantwortet und Fakten über die Familienmitglieder lernt. Außerdem kann er bspw. in Notfällen Alarm schlagen, da er zum Beispiel liegende Personen erkennt, oder bemerkt, wenn Personen auf Ansprache nicht mehr reagieren. 

### Use Cases, oder warum ein B-Bot in jeden Haushalt gehört: 
Viele Kinder und auch Erwachsene sind heute viel zu Hause und teilweise sehr einsam. Sie verbringen die Zeit mit Doom Scrolling und konsumieren Social Media Content und andere Medien. Aktuelle Assistenten beschränken sich hauptsächlich auf Frage und Antwortmöglichkeiten, oder schlicht auf die Bedienung des Smart Homes. 
Die Vision mit RON ist, dass ein echter Begleiter im Haushalt lebt, der mehrere Stunden jeden Tag den Familienalltag mit erlebt, besondere Ereignisse aufzeichnen kann ("Ron, mach ein Foto von uns") und durch seine ausgeprägte Memory Funktion eine echte Hilfe ist, wenn man sich nochmal an etwas erinnern möchte: 
- Ron, erinnere mich täglich an meine Pillen 
- Ron, ich muss die roten Pillen montags nehmen und die grünen dienstags
- Ron, welche Pille muss ich heute nehmen? Ron schaut nach dem Wochentag und in seinen Erinnerungen und beantwortet wahrheitsgemäß die Frage
- Ron, schick eine Mail an meinen Steuerberater über den Tesla, den ich mir konfiguriert habe (Ron kann Zugang zu bspw. Chrome haben und den Verlauf des Browsers einsehen)
- Das Gesicht von Ron kann Ergebnisse aus Online Suchen zeigen und durch das Touch Display ist er als voll funktionsfähiger Rechner nutzbar, ohne Maus und Tastatur
- Ron kann mit weiteren IOT-Devices ausgestattet werden, um bspw. Feuer oder Kohlenmonoxid zu detektieren
- Ultrakurzdistanz Beamer können RON ermöglichen Filme und Fotos an die Wand zu projizieren (wie im Film), oder Bilder aus seinen Erinnerungen zu generieren (er hat die Gesichter der Personen, des Hauses, etc.). In der ersten Iteration können diese Erinnerungen erstmal auf dem Display angezeigt werden, um die grundsätzliche Machbarkeit zu überprüfen. Wenn eine dieser visuellen Erinnerungen erzeugt wird, dann können diese auch dauerhaft auf seiner Festplatte / Cloud abgelegt werden mit einem eindeutigen Verweis, sodass er diese immer wieder zeigen kann (bspw. RON guckt mit der Familie einen Film und "erinnert" sich an diese Situation und projeziert ein generiertes Bild davon)
- Ron kann im Haus nachts oder bei Abwesenheit Türen bewachen und bei ungewöhnlichen Ereignissen Push Notifications an die RON App schicken (dadurch macht man keine Hintertür an die Kamera auf, sondern wird lediglich benachrichtigt)
- wenn Kinder oder ältere Leute alleine im Haus sind, dann kann RON auf diese aufpassen und im Notfall Hilfe rufen (bspw. Ohnmächtig, gestürzt, o.ä.)
- wenn die Echtzeitverarbeitung schnell genug funktioniert (muss erprobt werden), dann wäre es denkbar mit Ron Filme zu gucken, oder Musik zu hören und sich danach mit ihm darüber zu unterhalten (RON kann bspw. kurze Abschnitte zusammenfassen und in seinem Repository speichern)
- RON kann durch das Haus patrouillieren und bspw. schauen, ob der Hundenapf noch Futter hat, das Meerschweinchen schmutzig ist, oder das Kinderzimmer aufgeräumt werden sollte, Waschmaschine oder Herd (noch) an ist. 
- Wenn man das Haus verlässt, kann RON einen immer daran erinnern, einen Schlüssel mitzunehmen

### Die Bedienung / RON OS
Im regulären Betrieb soll RON komplett per Sprache als AI Agent bedienbar sein. Er ist ein "Always-on"-Device, d.h. solange seine Batterie Strom hat, bzw. er lädt, soll er über ein Wake-Word aufgeweckt werden können. Außerdem soll er sich niemals abschalten, wenn er nicht darum explizit gebeten wird (es müssen Thresholds definiert werden, wenn bspw. der Akku zu niedrig ist, damit er sich auflädt oder darum bittet, o.ä.). Grundsätzlich ist jedoch ein Dauerbetrieb geplant. 
In der Prototypen Phase wird RON per Shell konfiguriert, sodass WLAN etc. darüber verbunden werden können. Im späteren Verlauf muss es ein RON OS geben, eine einfache Bedienoberfläche, die es dem Nutzer ermöglicht die Ersteinrichtung zu machen (Name, WLAN verbinden, ggf. Social Media Profile verbinden, etc.). RON OS ist ein Agentic AI OS
Haptische Eingabe soll nur das Fallback sein, der Kern der Bedienung ist, dass die volle Power der Natürlichsprachlichkeit gezeigt wird, sodass die Bedienung absolut einfach und intuitiv ist. 
Beispiel: 
- "Ron, lass uns das WLAN konfigurieren"
	- "Gern! Welches dieser WLANs ist deins? (nur die ersten 5 werden aufgezählt, davon ausgehend, dass gemäß Verbindungsstärke das Richtige dabei ist)"
- User nennt das WLAN
- RON fragt nach dem Passwort und über den Tool Call kann er selbstständig sein eigenes WLAN einrichten (Linux Shell, Config wie bei Arch Linux)
- Displayhelligkeit steuern 
- Bewegung (komm her, fahr in die Küche, such Mama, etc.)
- Online Abfragen / Wissensdatenbanken 
- Wie viel Akku hast du noch? 
- Ron, schick eine Mail (Mail Konfiguration ggf. per App später, oder direkt in RON OS?)
- Video Calls? Welche Plattform, außer Jitsi? WhatsApp Web als WebApp? 
- IOT-Devices 
	- wie voll ist der Kühlschrank, brauchen wir Milch? 
	- 3D Drucker checken 
- Ron, mach Notizen 
- Ron, mach Kalendereinträge

### The Brain
Das Gehirn von Ron ist ein Nvidia Jetson Orin 8GB der ersten Generation, mindestens in der aktuellen Prototypen Phase und der 1. Generation. Das Gehirn besteht aus: 
- lokalen LLMs für direkte Verarbeitung der Anfrage
- Cloud LLMs via API, um Wissen aus dem Internet abzufragen
- einem langfristigen Gedächtnis, hierfür soll RON regelmäßig sein Gehirn in einem online (Git) Repository ablegen, sodass er immer ein Backup automatisch hat. Das Konzept der genauen Erinnerung, oder wie Erinnerungsbausteine abgelegt werden sollen, muss noch ermittelt werden. Ziel ist, dass RON sich, wie auch immer, auch an Ereignisse erinnern kann, die 1 oder 2 Jahre in der Vergangenheit liegen (evtl. ist hier ein RAG Ansatz sinnvoll)

Die Augen von RON sind ein AI-basiertes Kamerasystem, dass Winkel, Personen, Gesten und Distanzen erkennen und messen kann. Dadurch kann er bspw. auf Winken reagieren, erkennen, ob jemand traurig oder wütend ist und sich auch zu Personen hin- oder von ihnen abwenden. 

Das Gesicht wird auf einem 10" HD-Display mit Touchfunktion gerendert. Um eine möglichst flüssige Animation zu gewährleisten, soll das Gesicht mit 60 FPS dargestellt werden, sodass RON: 
- lächeln, 
- kauen, 
- lachen,
- traurig, 
- wütend,
- überrascht
schauen kann. Außerdem soll er wenn immer möglich bei Interaktionen Blickkontakt herstellen, um eine möglichst natürliche Kommunikation zu gewährleisten und Vertrauen aufzubauen. Die Oberfläche wird mit QT6 programmiert. Damit soll das Gesicht, als auch das RON OS dargestellt werden. 

Seine Größe wird 60-80cm betragen

### Vernetzung
In dem Film ist der B-Bot ein "bester Freund out-of-the-box". Er ist ein Device für Kinder und kann per Biometrie alle vorhandenen Informationen über den User aus dem B-Bot Netzwerk laden. Grundsätzlich wäre es denkbar, dass man RON mit LinkedIn oder anderen Social Networks verbindet, wenn über API möglich, damit erste Informationen über den User vorliegen. 
Außerdem ist es möglich über Open Source Frameworks RON Zugang zum Smarthome zu ermöglichen und somit Alexa und Co vollständig abzulösen. 