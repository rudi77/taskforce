using Configuration;
using Taskforce.Core;
using Taskforce.Core.Strategy;

namespace App
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            //var config = TaskforceConfig.Create("./sample/taskforce_invoice.yaml");
            //var config = TaskforceConfig.Create("./sample/taskforce_fxrate.yaml");
            //var config = TaskforceConfig.Create("./sample/taskforce_invoice2.yaml");
            var config = TaskforceConfig.Create("./sample/taskforce_receipt.yaml");
            //var receipts = new List<string> { @"C:\Users\rudi\Documents\297581.png" };
            var receipts = new List<string> { @"C:\Users\rudi\Documents\Arbeit\CSS\297595.jpeg.png" };

           
            var planner = new Planner(new OpenAIAssistantClient(), new ChainOfThoughtStrategy())
            {
                GeneralInstruction = config.PlanningConfig.GeneralInstruction,
                AnswerInstruction = config.PlanningConfig.AnswerInstruction
            };

            var noPlanPlanner = new NoPlanPlanner();
            var shortTermMemory = new ShortTermMemory();

            var agent = new Agent(new OpenAIAssistantClient(), planning: planner, shortTermMemory: shortTermMemory)
            {
                Role = config.AgentConfigs[0].Role,
                Mission = config.AgentConfigs[0].Mission,
            };

            // execute mission
            var response = await agent.ExecuteAsync(Query(), string.Empty, receipts);

            await Console.Out.WriteLineAsync("Final response:\n" + response);

            return;
        }

        static string Query()
        {
            //var query = "User: Extract all FX rates from the invoice and return them as a JSON object. ";
            //var query = "User: Extract all relevant invoice details from the given OCR result and provide json. Use the json format as defined.";
            var query = "User: Extract all relevant receipt details from the uploaded receipt image";

            return query;
        }

        static string Content()
        {
            return @"
                Invoice:
                RECHNUNG
                Handelsagentur Fux
                DATUM: 25.03.2020
                Rechnung Nr.: 1954746731
                KUNDEN-ID: HVK1A
                Schwarzstraße 45 5020 Salzburg
                RECHNUNGSADRESSE LIEFERADRESSE
                Massimo Mustermann
                Match GmbH
                Bergheimerstraße 14
                5020 Salzburg
                +436608553947
                Rechnungsadresse
                Bestellnummer: 258934 Bestelldatum: 15.3.2020
                Auftragsnummer: A1237B Auftragsdatum: 15.3.2020
                BESCHREIBUNG
                Menge Steuersatz Preis (netto)
                Lieferdatum: 20.3.2020 Lieferscheinnummer: LS185
                Steinway Konzert Flügel Weiß 1 20 % 499 000.00 €
                Dirigierstab Elfenbein 1 20 % 780.00 €
                Lieferdatum: 22.3.2020 Lieferscheinnummer: LS187
                nVidia GPU M60 'Tesla'
                4 20 % 28 560.00 €
                Mars Riegel
                1000 10 % 800.00 €
                Gesamtbetrag netto 529 140.00 €
                10 % 20 %
                Steuerbetrag 80.00 € 105 668.00 € 105 748.00 €
                Netto Betrag
                800.00 €
                528 340.00 € 529 140.00 €
                Summe brutto 880.00 € 634 008.00 € 634 888.00 €
                Zahlung: innerhalb von 10 Tagen 2 % Skonto
                30 Tage netto
                Alle Zahlungen an Handelsagentur Fux
            ";
        }

        static string Content2()
        {
            return @"                
                Pozice
                Jednotka Cena bez DPH DPH
                DPH Cena s DPH
                VGM
                VGM
                Port Doc
                B/L
                OF
                USD 123.25 to 0.9243
                PICK UP
                CZK 4500 to 0.0432
                NESTOH
                USD 90 to 0.9243
                Sub Total
                1
                1
                1
                1
                1
                1
                1
                4,93 4,93 21 %
                10,00 10,00 21 %
                10,00 10,00 21 %
                35,00 35,00 21 %
                113,92 113,92 21 %
                1,04 5,97
                2,10 12,10
                2,10 12,10
                7,35 42,35
                23,92 137,84
                194,40 194,40 21 %
                40,82 235,22
                83,19 83,19 21 %
                17,47 100,66
                94,80 546,24
                451,44
                Cástka k úhrade: EUR 546,24
                Celkem bez DPH
                DPH
                DPH 21 %
                CZK
                10.611,03
                2.228,32
                Cástka k úhrade
                CZK
                10.611,03
                2.228,32               
            ";
        }

        static string Content3()
        {
            return @"
2|7 3|bluecue
4|digital 5|strategies
6|bluecue 7|consulting 8|GmbH 9|& 10|Co. 11|KG 12| 13|August 14|Schroeder-Straße 15|4 16| 17|33602 18|Bielefeld
33|Phone
34|49 35|521 36|92279800 37| 38|Fax: 39|+ 40|49 41|521 42|32901351 43| 44|E-Mail: 45|info@bluecue.de
19|DACHSER 20|SE
21|Corporate 22|IT
23|Herr 24|Thomas 25|Kluge
26|Thomas 27|Dachser-Str. 28|2
61|Seite:
62|Auftrag-Nr.:
63|Datum :
64|Ihr Auftrag:
65|Ihr Auftrag 66|vom:
67|Kunden-Nr.:
68|Ansprechpartner:
29|D 30|87439 31|Kempten
69|1 70|- 71|2
72|AB10003038
73|08.08.2022
74|4500066016
75|05.08.2022
76|40314
77|Inga 78|Knoche
32|AUFTRAG
46|Wir 47|bedanken 48|uns 49|für 50|Ihren 51|Auftrag, 52|den 53|wir 54|hiermit 55|wie 56|folgt 57|bestätigen.
60|Artikelcode/Bezeichnung
80|Gesamt
58|Pos 59|Anzahl
79|Einzelpreis
84|Angebot 85|Kunde: 86|AN10005790 87|vom 88|01.08.2022
89|Ihre 90|Bestellung Nr.: 91|4500066016 92|vom 93|05.08.2022
81|1.1 82|1 83|Stück
94|bluecue 95|Review 96|for 97|NetApp 98|Add-on 99|12 100|mths 101|2.500,00
152|2.500,00
102|Wartungsverlägerung 103|um 104|12 105|Monate 106|ab 107|08.2022
108|bluecue 109|Review 110|for 111|NetApp 112|ist 113|ein 114|leistungsfähiges 115|Modul 116|für 117|Analyse
118|und 119|Auswertung 120|von 121|Ordnern 122|Files 123|und 124|ACL-Objekten.
125|Die 126|Datenhaltung 127|basiert 128|auf 129|Splunk 130|Enterprise 131|und 132|setzt 133|eine
134|entsprechende 135|Lizenz 136|voraus.
137|Diese 138|Wartungsverlägerung 139|erlaubt 140|jeweils 141|die 142|Nutzung 143|der 144|aktuellsten
145|Version 146|von 147|bluecue 148|Review 149|for 150|NetApp.
151|****
153|Zwischensumme
154|EUR 155|2.500,00
156|w 157|w 158|w. 159|b 160|l 161|u 162|e 163|c 164|u 165|e 166|. 167|d 168|e
169|bluecue 170|consulting 171|GmbH 172|& 173|Co. 174|KG 175|. 176|Sitz 177|Bielefeld 178|Amtsgericht-Registergericht 179|Bielefeld 180| 181|HRA 182|15928
183|Persönlich 184|haftende 185|Gesellschafterin: 186|bluecue 187|consulting 188|Beteiligungsgesellschaft 189|mbH 190| 191|Sitz 192|Bielefeld 193| 194|Amtsgericht- 195|Registergericht- 196|Bielefeld 197| 198|HRB 199|40940
200|Geschäftsführer: 201|Jovan 202|Ilic 203|Nico 204|Lüdemann 205|und 206|Mark 207|Schönrock 208| 209|Umsatzsteuer-Identifikationsnummer 210|gemäß 211| 212|27a 213|Umsatzsteuergesetz: 214|DE 287 832 849
215|
216|7 217|bluecue
218|digital 219|strategies
220|Auftrags-Nr. 221|AB10003038 222|vom 223|08.08.2022 224|Seite: 225|2 226|- 227|2
228|Pos 229|Anzahl 230|Artikelcode/Bezeichnung 231|Einzelpreis 232|Gesamt
233|Übertrag 234|2.500,00
235|Netto 236|- 237|Betrag 238|EUR 239|2.500,00
240|+ 241|19,00 242|% 243|Ust 244|EUR 245|475,00
246|Brutto-Betrag
247|EUR 248|2.975,00
249|Gültigkeit 250|Lieferzeit 251|und 252|AGB
253|Zahlungskonditionen 254|: 255|14 256|Tage 257|netto
258|Es 259|gelten 260|unsere 261|Allgemeinen 262|Geschäftsbedigungen 263|- 264|http://www.bluecue.de/agb/
265|Die 266|Ware 267|bleibt 268|bis 269|zur 270|vollständigen 271|Bezahlung 272|unser 273|Eigentum.
274|Für 275|Rückfragen 276|und 277|eventuell 278|weitere 279|Informationen
280|stehen 281|wir 282|Ihnen 283|jederzeit 284|gern 285|zur 286|Verfügung.
287|Mit 288|freundlichen 289|Grüßen
290|Inga 291|Knoche
292|w 293|w 294|w- 295|b 296|l 297|u 298|e 299|c 300|u 301|e. 302|d 303|e
304|bluecue 305|consulting 306|GmbH 307|& 308|Co. 309|KG 310| 311|Sitz 312|Bielefeld 313| 314|Amtsgericht- 315|Registergericht- 316|Bielefeld 317| 318|HRA 319|15928
320|Persönlich 321|haftende 322|Gesellschafterin: 323|bluecue 324|consulting 325|Beteiligungsgesellschaft 326|mbH 327| 328|Sitz 329|Bielefeld 330| 331|Amtsgericht- 332|Registergericht- 333|Bielefeld 334| 335|HRB 336|40940
337|Geschäftsführer: 338|Jovan 339|Ilic 340|Nico 341|Lüdemann 342|und 343|Mark 344|Schönrock 345| 346|Umsatzsteuer-Identifikationsnummer 347|gemäß 348| 349|27a 350|Umsatzsteuergesetz: 351|DE 287 832 849
            ";
        }

        static string Content4()
        {
            return @"
1|/Eh 2|ASAM 3|E 4|R
5|HARTGESTEINWERK 6|WANKO
7|Gebrüder 8|Haider 9|Bauunternehmung 10|GmbH
11|Großraming 12|40
13|4463 14|Großraming
34|Rechnung 35|A/21/020418
36|Seite 37|1
15|UID-Nr.: 16|ATU61382248
17|Kunde: 18|214451
19|Baustelle: 20|0297 21|Dürnrohr 22|EVN 23|Loco 24|2750
25|Lieferscheine 26|über 27|Zeitraum: 28|16.06.21 29|- 30|18.06.21
38|Rech.-Datum
39|23.06.21
40|Claudia 41|Gugglberger 42|05 43|/ 44|0799-3255, 45|C.Gugglberger@asamer.at
31|Datum 32|Lieferschein 33|Bezeichnung
46|Menge
47|Rab 48|% 49|Preis 50|Netto 51|Betrag 52|Netto
53|16.06.21 54|1241418
89|KK 90|0/90 91|U9
92|Roadpricing
93|KK 94|0/90 95|U9
96|Roadpricing
97|KK 98|0/90 99|U6
100|Roadpricing
101|KK 102|0/90 103|U6
104|Roadpricing
105|KK 106|0/90 107|U9
108|Roadpricing
109|KK 110|0/90 111|U9
112|Roadpricing
113|KK 114|0/90 115|U9
116|Roadpricing
117|KK 118|0/90 119|U6
120|Roadpricing
121|KK 122|0/90 123|U9
124|Roadpricing
125|KK 126|0/90 127|U9
128|Roadpricing
129|KK 130|0/90 131|U9
132|Roadpricing
133|KK 134|0/90 135|U9
136|Roadpricing
137|KK 138|0/90 139|U9
140|Roadpricing
141|KK 142|0/90 143|U9
144|Roadpricing
145|KK 146|0/90 147|U9
148|Roadpricing
149|KK 150|0/90 151|U9
152|Roadpricing
153|KK 154|0/90 155|U9
156|Roadpricing
157|KK 158|0/90 159|U9
160|Roadpricing
203|26,94 204|TO
205|1,00 206|PA
207|25,88 208|TO
209|1,00 210|PA
211|27,60 212|TO
213|1,00 214|PA
215|27,00 216|TO
217|1,00 218|PA
219|25,76 220|TO
221|1,00 222|PA
223|23,28 224|TO
225|1,00 226|PA
227|26,56 228|TO
229|1,00 230|PA
231|28,72 232|TO
233|1,00 234|PA
235|27,14 236|TO
237|1,00 238|PA
239|24,40 240|TO
241|1,00 242|PA
243|26,68 244|TO
245|1,00 246|PA
247|27,58 248|TO
249|1,00 250|PA
251|25,60 252|TO
253|1,00 254|PA
255|25,02 256|TO
257|1,00 258|PA
259|27,30 260|TO
261|1,00 262|PA
263|24,94 264|TO
265|1,00 266|PA
267|26,56 268|TO
269|1,00 270|PA
271|25,38 272|TO
273|1,00 274|PA
283|7,05 284|189,93
285|0,00 286|0,00
287|7,05 288|182,45
289|0,00 290|0,00
291|7,05 292|194,58
293|0,00 294|0,00
295|7,05 296|190,35
297|0,00 298|0,00
299|7,05 300|181,61
301|0,00 302|0,00
303|7,05 304|164,12
305|0,00 306|0,00
307|7,05 308|187,25
309|0,00 310|0,00
311|7,05 312|202,48
313|0,00 314|0,00
315|7,05 316|191,34
317|0,00 318|0,00
319|7,05 320|172,02
321|0,00 322|0,00
323|7,05 324|188,09
325|0,00 326|0,00
327|7,05 328|194,44
329|0,00 330|0,00
331|7,05 332|180,48
333|0,00 334|0,00
335|7,05 336|176,39
337|0,00 338|0,00
339|7,05 340|192,47
341|0,00 342|0,00
343|7,05 344|175,83
345|0,00 346|0,00
347|7,05 348|187,25
349|0,00 350|0,00
351|7,05 352|178,93
353|0,00 354|0,00
355|Übertrag 356|. 357|. 358|. 359|. 360|. 361|. 362|3.330,01
55|16.06.21 56|1241450
57|16.06.21 58|1241505
59|17.06.21 60|1241517
61|17.06.21 62|1241520
63|17.06.21 64|1241527
65|17.06.21 66|1241529
67|17.06.21 68|1241552
69|17.06.21 70|1241561
71|17.06.21 72|1241563
73|17.06.21 74|1241565
75|17.06.21 76|1241591
77|17.06.21 78|1241592
79|17.06.21 80|1241594
81|17.06.21 82|1241610
83|17.06.21 84|1241614
85|17.06.21 86|1241615
87|17.06.21 88|1241618
161|Asamer 162|Kies- 163|und 164|Betonwerke 165|GmbH
166|Unterthalham 167|Straße 168|2
169|4694 170|Ohlsdorf 171|? 172|Österreich
173|T 174|+43 175|(0)5 176|0799-0 177| 178|F 179|DW 180|1005
181|E 182|office@asamer.at
183|Hartgesteinwerk 184|Wanko
185|Schlossstraße 186|19
187|3508 188|Meidling 189|im 190|Tal 191| 192|Österreich
193|T 194|+43 195|(0)5 196|0799-3700 197| 198|F 199|DW 200|3705
201|E 202|office@asamer.at
275|Firmendaten
276|ATU 22073306
277|DVR 278|0092428/010480
279|Firmenbuch 280|Wels
281|FN 282|107110s
363|Bankverbindung
364|Oberbank 365|AG
366|IBAN 367|AT39 1500 0002 8158 3716
368|BIC 369|OBKLAT2L
370|www.asamer.at
371|/Eh 372|ASAM 373|E 374|R
375|HARTGESTEINWERK 376|WANKO
377|Rechnung 378|A/21/020418
393|Seite 394|2
379|UID-Nr.:
380|Kunde:
381|Baustelle:
382|ATU61382248
383|214451
384|0297 385|Dürnrohr 386|EVN 387|Loco 388|2750
395|Rech.-Datum
396|23.06.21
389|Datum 390|Lieferschein 391|Bezeichnung
392|Menge
397|Rab 398|% 399|Preis 400|Netto 401|Betrag 402|Netto
670|Übertrag. 671|. 672|. 673|. 674|. 675|. 676|.
734|3.330,01
735|185,13
736|0,00
737|185,98
738|0,00
739|176,39
740|0,00
741|190,49
742|0,00
743|189,36
744|0,00
745|191,20
746|0,00
747|193,88
748|0,00
749|182,03
750|0,00
751|166,80
752|0,00
753|197,54
754|0,00
755|196,27
756|0,00
757|195,29
758|0,00
759|185,27
760|0,00
761|184,01
762|0,00
763|179,35
764|0,00
765|192,89
766|0,00
767|181,89
768|0,00
769|182,88
770|0,00
771|192,75
772|0,00
773|188,66
774|0,00
775|186,54
776|0,00
777|187,81
778|7.442,42
403|17.06.21 404|1241624 405|KK 406|0/90 407|U9
408|Roadpricing
409|17.06.21 410|1241629 411|KK 412|0/90 413|U9
414|Roadpricing
415|17.06.21 416|1241638 417|KK 418|0/90 419|U9
420|Roadpricing
421|17.06.21 422|1241639 423|KK 424|0/90 425|U9
426|Roadpricing
427|17.06.21 428|1241642 429|KK 430|0/90 431|U9
432|Roadpricing
433|17.06.21 434|1241655 435|KK 436|0/90 437|U9
438|Roadpricing
439|17.06.21 440|1241656 441|KK 442|0/90 443|U9
444|Roadpricing
445|17.06.21 446|1241667 447|KK 448|0/90 449|U9
450|Roadpricing
451|18.06.21 452|1241683 453|KK 454|0/90 455|U9
456|Roadpricing
457|18.06.21 458|1241689 459|KK 460|0/90 461|U9
462|Roadpricing
463|18.06.21 464|1241690 465|KK 466|0/90 467|U9
468|Roadpricing
469|18.06.21 470|1241692 471|KK 472|0/90 473|U9
474|Roadpricing
475|18.06.21 476|1241698 477|KK 478|0/90 479|U6
480|Roadpricing
481|18.06.21 482|1241700 483|KK 484|0/90 485|U9
486|Roadpricing
487|18.06.21 488|1241717 489|KK 490|0/90 491|U9
492|Roadpricing
493|18.06.21 494|1241728 495|KK 496|0/90 497|U9
498|Roadpricing
499|18.06.21 500|1241733 501|KK 502|0/90 503|U6
504|Roadpricing
505|18.06.21 506|1241737 507|KK 508|0/90 509|U9
510|Roadpricing
511|18.06.21 512|1241743 513|KK 514|0/90 515|U9
516|Roadpricing
517|18.06.21 518|1241744 519|KK 520|0/90 521|U9
522|Roadpricing
523|18.06.21 524|1241756 525|KK 526|0/45 527|U9
528|Roadpricing
529|18.06.21 530|1241772 531|KK 532|0/45 533|U9
619|TO
620|PA
621|TO
622|PA
623|TO
624|PA
625|TO
626|PA
627|TO
628|PA
629|TO
630|PA
631|TO
632|PA
633|TO
634|PA
635|TO
636|PA
637|TO
638|PA
639|TO
640|PA
641|TO
642|PA
643|TO
644|PA
645|TO
646|PA
647|TO
648|PA
649|TO
650|PA
651|TO
652|PA
653|TO
654|PA
655|TO
656|PA
657|TO
658|PA
659|TO
660|PA
661|TO
576|26,26
577|1,00
578|26,38
579|1,00
580|25,02
581|1,00
582|27,02
583|1,00
584|26,86
585|1,00
586|27,12
587|1,00
588|27,50
589|1,00
590|25,82
591|1,00
592|23,66
593|1,00
594|28,02
595|1,00
596|27,84
597|1,00
598|27,70
599|1,00
600|26,28
601|1,00
602|26,10
603|1,00
604|25,44
605|1,00
606|27,36
607|1,00
608|25,80
609|1,00
610|25,94
611|1,00
612|27,34
613|1,00
614|26,76
615|1,00
616|26,46
617|1,00
618|26,64
677|7,05
678|0,00
679|7,05
680|0,00
681|7,05
682|0,00
683|7,05
684|0,00
685|7,05
686|0,00
687|7,05
688|0,00
689|7,05
690|0,00
691|7,05
692|0,00
693|7,05
694|0,00
695|7,05
696|0,00
697|7,05
698|0,00
699|7,05
700|0,00
701|7,05
702|0,00
703|7,05
704|0,00
705|7,05
706|0,00
707|7,05
708|0,00
709|7,05
710|0,00
711|7,05
712|0,00
713|7,05
714|0,00
715|7,05
716|0,00
717|7,05
718|0,00
719|7,05
720|Übertrag. 721|. 722|. 723|. 724|. 725|.
534|Asamer 535|Kies- 536|und 537|Betonwerke 538|GmbH
539|Unterthalham 540|Straße 541|2
542|4694 543|Ohlsdorf 544|? 545|Österreich
546|T 547|+43 548|(0)5 549|0799-0 550| 551|F 552|DW 553|1005
554|E 555|office@asamer.at
556|Hartgesteinwerk 557|Wanko
558|Schlossstraße 559|19
560|3508 561|Meidling 562|im 563|Tal 564| 565|Österreich
566|T 567|+43 568|(0)5 569|0799-3700 570| 571|F 572|DW 573|3705
574|E 575|office@asamer.at
662|Firmendaten
663|ATU 22073306
664|DVR 665|0092428/010480
666|Firmenbuch 667|Wels
668|FN 669|107110s
726|Bankverbindung
727|Oberbank 728|AG
729|IBAN 730|AT39 1500 0002 8158 3716
731|BIC 732|OBKLAT2L
733|www.asamer.at
779|/Eh 780|ASAM 781|E 782|R
783|HARTGESTEINWERK 784|WANKO
795|Rechnung 796|A/21/020418
801|Seite 802|3
803|Rech.-Datum
804|23.06.21
785|UID-Nr.:
786|Kunde:
787|Baustelle:
788|ATU61382248
789|214451
790|0297 791|Dürnrohr 792|EVN 793|Loco 794|2750
797|Datum 798|Lieferschein 799|Bezeichnung 800|Menge
805|Rab 806|% 807|Preis 808|Netto 809|Betrag 810|Netto
854|Übertrag. 855|. 856|. 857|. 858|. 859|. 860|.
870|7.442,42
871|0,00
872|182,60
873|0,00
874|166,38
875|0,00
876|171,17
877|0,00
878|192,32
879|0,00
811|Roadpricing
812|18.06.21 813|1241784 814|KK 815|0/45 816|U9
817|Roadpricing
818|18.06.21 819|1241788 820|KK 821|0/45 822|U9
823|Roadpricing
824|18.06.21 825|1241789 826|KK 827|0/90 828|U9
829|Roadpricing
830|18.06.21 831|1241809 832|KK 833|0/45 834|U9
835|Roadpricing
836|1,00 837|PA
838|25,90 839|TO
840|1,00 841|PA
842|23,60 843|TO
844|1,00 845|PA
846|24,28 847|TO
848|1,00 849|PA
850|27,28 851|TO
852|1,00 853|PA
861|0,00
862|7,05
863|0,00
864|7,05
865|0,00
866|7,05
867|0,00
868|7,05
869|0,00
880|Sammelpositionen:
888|Landschaftschutzabg. 889|NÖ 890|1.156,72 891|TO
911|0,217
912|EUR
913|EUR
914|EUR
915|Menge
916|891,44 917|TO
918|129,88 919|TO
920|135,40 921|TO
1046|250,99
1047|8.405,88
1048|1.681,18
1049|10.087,06
906|Nettosumme:
907|20 908|% 909|MwSt.
910|Bruttosumme:
881|Artikelsumme:
892|Artikel Nr.
893|AM0063W
894|AM0045W
895|AM0090W
896|Bezeichnung
897|KK 898|0/90 899|U9
900|KK 901|0/45 902|U9
903|KK 904|0/90 905|U6
927|MwSt. 928|-Betrag
929|1.681,18
930|1.681,18
882|MwSt. 883|%
884|20
885|Gesamt
922|MwSt 923|Bem 924|-Grundlage
925|8.405,88
926|8.405,88
886|Zahlungsbedingungen
931|21 932|Tage 933|3 934|% 935|Skonto 936|30 937|Tage 938|ohne 939|Abzug 940|berechnet 941|ab 942|23.06.21
943|Zahlung 944|per 945|23.07.21 946|Zahlbetrag: 947|10.087,06 948|EUR
949|Bei 950|Zahlung 951|vor 952|14.07.21 953|wird 954|ein 955|Skonto 956|von 957|302,61 958|EUR 959|gewährt 960|Zahlbetrag: 961|9.784,45 962|EUR
887|Lieferbedingungen
963|Lieferung 964|frei 965|Bau
966|Geben 967|Sie 968|bitte 969|bei 970|der 971|Überweisung 972|im 973|Feld 974|KUNDENDATEN 975|folgende Nummer 976|ein: 977|000021020418
978|oder 979|verwenden 980|Sie 981|den 982|QR 983|Code 984|für 985|Ihre 986|elektronische 987|Überweisung.
988|Asamer 989|Kies- 990|und 991|Betonwerke 992|GmbH
993|Unterthalham 994|Straße 995|2
996|4694 997|Ohlsdorf 998|? 999|Österreich
1000|T 1001|+43 1002|(0)5 1003|0799-0 1004| 1005|F 1006|DW 1007|1005
1008|E 1009|office@asamer.at
1010|Hartgesteinwerk 1011|Wanko
1012|Schlossstraße 1013|19
1014|3508 1015|Meidling 1016|im 1017|Tal 1018| 1019|Österreich
1020|T 1021|+43 1022|(0)5 1023|0799-3700 1024| 1025|F 1026|DW 1027|3705
1028|E 1029|office@asamer.at
1030|Firmendaten
1031|ATU 22073306
1032|DVR 1033|0092428/010480
1034|Firmenbuch 1035|Wels
1036|FN 1037|107110s
1038|Bankverbindung
1039|Oberbank 1040|AG
1041|IBAN 1042|AT39 1500 0002 8158 3716
1043|BIC 1044|OBKLAT2L
1045|www.asamer.at
            ";
        }
    }
}