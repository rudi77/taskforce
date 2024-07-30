using Taskforce.Abstractions;
using Taskforce.Configuration;
using Taskforce.Core;
using Taskforce.Core.Strategy;

internal class Program
{
    static async Task Main(string[] args)
    {
        var receipts = new List<string> { @"C:\Users\rudi\source\repos\receipt-gen\invoice.png" };
        var config = TaskforceConfig.Create("./sample/taskforce_lineitems.yaml");

        var agent1 = CreateAgent(config.PlanningConfig, config.AgentConfigs[0]);
        var agent2 = CreateAgent(config.PlanningConfig, config.AgentConfigs[1]);

        var pipeline = new AgentPipeline();
        pipeline.AddAgent(agent1);
        pipeline.AddAgent(agent2);

        var response = await pipeline.ExecuteAsync(Query(), string.Empty, receipts);

        await Console.Out.WriteLineAsync("Final response:\n" + response);
    }

    static Agent CreateAgent(PlanningConfig planningConfig, AgentConfig agentConfig)
    {
        var planner = new Planner(
            new OpenAIAssistantClient(),
            new ChainOfThoughtStrategy(),
            planningConfig);

        var noPlanPlanner = new NoPlanPlanner();

        var shortTermMemory = new ShortTermMemory();
        var agent = new Agent(
            new OpenAIAssistantClient(),
            shortTermMemory: shortTermMemory,
            planning: noPlanPlanner,
            config: agentConfig);

        return agent;
    }

    static string Query()
    {
        return "User: Extract all relevant receipt details from the uploaded receipt image";
    }
}
