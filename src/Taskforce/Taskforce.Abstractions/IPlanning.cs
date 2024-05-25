namespace Taskforce.Abstractions
{
    public interface IPlanning
    {
        /// <summary>
        /// Takes a user's question and tries it down it smaller pieces
        /// </summary>
        /// <param name="userPrompt"></param>
        /// <returns></returns>
        Task<List<string>> PlanAsync(string userPrompt);
    }
}