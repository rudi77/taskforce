using Taskforce.Abstractions;

namespace Planning
{
    public class Planner : IPlanning
    {
        private readonly ILLM _llm;

        public Planner(ILLM llm)
        {
            _llm = llm ?? throw new ArgumentNullException(nameof(llm));
        }

        public string GeneralInstruction { get; set; }

        public string AnswerInstruction { get; set; }

        public async Task<List<string>> PlanAsync(string userPrompt)
        {
            // TODO: Exception handling.

            var questionAnswerPromptPart = UserQuestion(userPrompt) + "\n" + AnswerInstruction;
            var response = await  _llm.SendMessageAsync(GeneralInstruction, questionAnswerPromptPart);
            var questions = Newtonsoft.Json.JsonConvert.DeserializeObject<Questions>(response.ToString());

            return questions.SubQuestions;
        }

        private static string UserQuestion(string userPrompt) {
            return "USER QUESTION" + "\n" + userPrompt;
        }
    }
}
