using Taskforce.Domain.Entities;
using Taskforce.Domain.Interfaces;

namespace Taskforce.Domain.Strategy
{
    public class ChainOfThoughtStrategy : IPlanningStrategy
    {
        public async Task<List<string>> PlanAsync(string userPrompt, IChatCompletion llm, string generalInstruction, string answerInstruction)
        {
            var questionAnswerPromptPart = $"USER QUESTION\n{userPrompt}\n{answerInstruction}";
            var response = await llm.SendMessageAsync(generalInstruction, questionAnswerPromptPart);
            var questions = Newtonsoft.Json.JsonConvert.DeserializeObject<Questions>(response.ToString());

            return questions.SubQuestions;
        }

        public async Task<List<string>> PlanAsync(string userPrompt, IList<byte[]> images, IChatCompletion llm, string generalInstruction, string answerInstruction)
        {
            var questionAnswerPromptPart = $"USER QUESTION\n{userPrompt}\n{answerInstruction}";
            var response = await llm.SendMessageAsync(generalInstruction, questionAnswerPromptPart, images);
            var questions = Newtonsoft.Json.JsonConvert.DeserializeObject<Questions>(response.ToString());

            return questions.SubQuestions;
        }
    }

}
