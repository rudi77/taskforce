using Taskforce.Abstractions;

namespace Planning.Strategy
{
    public class ChainOfThoughtStrategy : IPlanningStrategy
    {
        public async Task<List<string>> PlanAsync(string userPrompt, ILLM llm, string generalInstruction, string answerInstruction)
        {
            var questionAnswerPromptPart = $"USER QUESTION\n{userPrompt}\n{answerInstruction}";
            var response = await llm.SendMessageAsync(generalInstruction, questionAnswerPromptPart);
            var questions = Newtonsoft.Json.JsonConvert.DeserializeObject<Questions>(response.ToString());

            return questions.SubQuestions;
        }

        public async Task<List<string>> PlanAsync(string userPrompt, IList<string> imagePaths, ILLM llm, string generalInstruction, string answerInstruction)
        {
            var questionAnswerPromptPart = $"USER QUESTION\n{userPrompt}\n{answerInstruction}";
            var response = await llm.SendMessageAsync(generalInstruction, questionAnswerPromptPart, imagePaths);
            var questions = Newtonsoft.Json.JsonConvert.DeserializeObject<Questions>(response.ToString());

            return questions.SubQuestions;
        }
    }

}
