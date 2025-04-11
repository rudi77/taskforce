using Taskforce.Domain.Entities;
using Taskforce.Domain.Interfaces;

//must be removed
using Taskforce.Infrastructure.Observability;

namespace Taskforce.Domain.Strategy
{
    public class ChainOfThoughtStrategy : IPlanningStrategy
    {
        public async Task<List<string>> PlanAsync(string userPrompt, IChatCompletion llm, string generalInstruction, string answerInstruction)
        {
            var questionAnswerPromptPart = $"USER QUESTION\n{userPrompt}\n{answerInstruction}";
            var response = await llm.SendMessageAsync(generalInstruction, questionAnswerPromptPart);
            var questions = Newtonsoft.Json.JsonConvert.DeserializeObject<Questions>(response.ToString()) ?? new Questions();

            questions.SubQuestions.Add(userPrompt);

            return questions.SubQuestions;
        }

        public async Task<List<string>> PlanAsync(string userPrompt, IList<byte[]> images, IChatCompletion llm, string generalInstruction, string answerInstruction)
        {
            var questionAnswerPromptPart = $"USER QUESTION\n{userPrompt}\n{answerInstruction}";
            var response = await llm.SendMessageAsync(generalInstruction, questionAnswerPromptPart, images);


            Console.Out.WritePlannerLineAsync(response.ToString());

            var questions = Newtonsoft.Json.JsonConvert.DeserializeObject<Questions>(response.ToString()) ?? new Questions();

            questions.SubQuestions.Add(userPrompt);

            return questions.SubQuestions;
        }
    }

}
