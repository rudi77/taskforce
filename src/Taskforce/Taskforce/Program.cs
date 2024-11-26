using Microsoft.Extensions.Logging;
using Taskforce.Application;
using Taskforce.Configuration;
using Taskforce.Core.Entities;
using Taskforce.Domain.Entities;
using Taskforce.Domain.Interfaces;
using Taskforce.Domain.Services;
using Taskforce.Infrastructure.LLM;

internal class Program
{
    private static readonly ILoggerFactory LoggerFactory = Microsoft.Extensions.Logging.LoggerFactory.Create(builder =>
    {
        builder.AddConsole(); // Adds the console logger
    });

    static async Task Main(string[] args)
    {
        var logger = LoggerFactory.CreateLogger("Program");
        logger.LogInformation("Application starting...");

        // Examp

        var receipts = new List<string> { @"C:\Users\rudi\Documents\Arbeit\CSS\297657.png" };
        List<byte[]> receipts_bytes = receipts.Select(File.ReadAllBytes).ToList();
        var config = TaskforceConfig.Create("./Configuration/sample/taskforce_receipt.yaml");

        var planner = new Planner(
            new OpenAIChatClient(),
            new NoPlanningStrategy(), //ChainOfThoughtStrategy(),
            config.PlanningConfig);

        var agent1 = CreateAgent(config.PlanningConfig, config.AgentConfigs[0], planner);
        var agent2 = CreateAgent(config.PlanningConfig, config.AgentConfigs[1], planner);

        var pipeline = new AgentPipeline();
        pipeline.AddAgent(agent1);
        pipeline.AddAgent(agent2);

        //var response = await pipeline.ExecuteAsync(Query(), Content(), receipts_bytes);
        var response = await pipeline.ExecuteAsync(Query(), "", receipts_bytes);
        await Console.Out.WriteLineAsync("Final response:\n" + response);
    }

    static Agent CreateAgent(PlanningConfig planningConfig, AgentConfig agentConfig, IPlanning planner)
    {
        var shortTermMemory = new ShortTermMemory();
        var agent = new Agent(
            llm: new OpenAIChatClient(),
            planning: planner,
            config: agentConfig,
            memoryManager: new MemoryManager(shortTermMemory),
            promptBuilder: new PromptBuilder(agentConfig),
            LoggerFactory.CreateLogger("Agent"));

        return agent;
    }

    static string Query()
    {
        return "User: Extract all relevant receipt details from the uploaded receipt image";
    }

    static string Content()
    {
        return @"
        ***
        CITYHOTEL
        Mir
        KURFÜRST
        t BALDUIN
        GARNI
        City Hotel Kurfürst Balduin GmbH  Hohenfelder Stoße 12  56068 Koblenz
        CSS AG
        Friedrich Dietz-Straße 1
        36093 Künzell
        CITY HOTEL
        KURFÜRST BALDUIN
        GMBH
        Hohenfelder Straße 1z
        56068 Koblenz
        Telefon 0261-1332-0
        Telefax 02 61-13 32-100
        Datum:
        Zimmer:
        Anreise:
        Abreise:
        Steuer-Nr.:
        Seite:
        31.08.2023
        407
        30.08.2023
        31.08.2023
        11/22/650/1220/9
        1/1
        Internet:
        www.cityhotel-koblenz.de
        E-Mail:
        info@cityhotel-koblenz.de
        Rechnung
        Rechnungsnummer 150513
        Kassierer :Mallmann, Jutta
        Herr Martin Waigand
        Datum Beschreibung
        Kredit ?
        Debit ?
        30.08.2023
        30.08.2023
        30.08.2023
        31.08.2023
        Garage Pauschale
        Frühstück
        Übernachtung
        Mastercard
        0,00
        0,00
        0,00
        99,80
        12,00
        13,80
        74,00
        0,00
        99,80
        Gesamt
        99,80
        0,00 ?
        Offener Saldo
        Diese Rechnung enthält folgende MwSt. -Beträge:
        Netto
        78,19 E
        13,56 E
        MWST
        5,47 E
        2,58 E
        Brutto
        83,66 ?
        16,14 ?
        MWST
        Tax 7 %
        Tax 19 %
        KassenSichV
        Transaktion Beginn Transaktion Ende
        Transaktionsnummer
        Seriennummer TSE
        22f7ac52dec415355d4a781795b50ad97 900326063
        aefcea04cad35fe15c5b041Ib6e140c
        31.08.2023 08:02:46 31.08.2023 08:02:47
        Wir danken für Ihren Besuch und wünschen eine angenehme Heimreise.
        Volksbank Koblenz
        Mittelrhein e.G.
        IBAN: DE63 5776 15911060 8o6o 00
        BIC: GENODE3IK0B
        Geschäftsführer: Steuer-Nr.
        Bankverbindungen:
        Sparkasse Koblenz
        IBAN: DE86 5705 0120 0000 0047 47
        BIC: MALADE51KOB
        11/22/650/1220/9 REGION M11TELRHEIN
        Hendrik Rooze
        Amtsgericht Koblenz USt-ID Nr.
        HRB 6431 DE 212 546 787
        ";
    }
}
