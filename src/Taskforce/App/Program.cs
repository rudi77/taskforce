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
            var config = TaskforceConfig.Create("./sample/taskforce_invoice.yaml");

            var llm = new OpenAILLM();

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
            var response = await agent.ExecuteAsync(Query(), Content());

            await Console.Out.WriteLineAsync(response);

            return;
        }

        static string Query()
        {
            var query = "User: Extract all contacts from the invoice and return them as a JSON object. Differentiate between invoice sender and receiver.";

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
    }
}