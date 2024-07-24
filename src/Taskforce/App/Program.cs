using Configuration;
using Memory;
using Planning;
using Taskforce.Agent;
using Taskforce.LLM;

namespace App
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            //var config = TaskforceConfig.Create("./sample/taskforce_invoice.yaml");
            var config = TaskforceConfig.Create("./sample/taskforce_fxrate.yaml");

            var llm = new OpenAIAssistantClient();

            var planner = new Planner(llm)
            {
                GeneralInstruction = config.PlanningConfig.GeneralInstruction,
                AnswerInstruction = config.PlanningConfig.AnswerInstruction
            };

            var shortTermMemory = new ShortTermMemory();

            var agent = new Agent(llm, planning: planner, shortTermMemory: shortTermMemory)
            {
                Role = config.AgentConfigs[0].Role,
                Mission = config.AgentConfigs[0].Mission,
            };

            // execute mission
            var response = await agent.ExecuteAsync(Query(), Content2());

            await Console.Out.WriteLineAsync("Final response:\n" + response);

            return;
        }

        static string Query()
        {
            var query = "User: Extract all FX rates from the invoice and return them as a JSON object. ";

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
    }
}