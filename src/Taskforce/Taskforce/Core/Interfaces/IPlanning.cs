namespace Taskforce.Domain.Interfaces
{
    public interface IPlanning
    {
        /// <summary>
        /// Takes a user's question and tries to break it down it smaller pieces
        /// </summary>
        /// <param name="userPrompt"></param>
        /// <returns></returns>
        Task<List<string>> PlanAsync(string userPrompt);

        /// <summary>
        /// 
        /// </summary>
        /// <param name="userPrompt"></param>
        /// <param name="images"></param>
        /// <returns></returns>
        Task<List<string>> PlanAsync(string userPrompt, IList<byte[]> images);
    }
}