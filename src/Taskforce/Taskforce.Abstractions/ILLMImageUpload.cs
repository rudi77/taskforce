namespace Taskforce.Abstractions
{
    /// <summary>
    /// 
    /// </summary>
    public interface ILLMImageUpload
    {
        /// <summary>
        /// Uploads images
        /// </summary>
        /// <param name="imagePaths"></param>
        /// <returns>A list of Image Ids</returns>
        Task<string[]> UploadFieAsync(IList<string> imagePaths);
    }
}