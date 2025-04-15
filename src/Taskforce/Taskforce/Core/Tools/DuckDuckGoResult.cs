namespace Taskforce.Core.Tools
{
    public class DuckDuckGoResult
    {
        public string Abstract { get; set; }
        public string AbstractSource { get; set; }
        public string AbstractText { get; set; }
        public string AbstractURL { get; set; }
        public string Answer { get; set; }
        public string AnswerType { get; set; }
        public string Definition { get; set; }
        public string DefinitionSource { get; set; }
        public string DefinitionURL { get; set; }
        public string Entity { get; set; }
        public string Heading { get; set; }
        public string Image { get; set; }
        public int ImageHeight { get; set; }
        public int ImageIsLogo { get; set; }
        public int ImageWidth { get; set; }
        public Infobox Infobox { get; set; }
        public string OfficialDomain { get; set; }
        public string OfficialWebsite { get; set; }
        public string Redirect { get; set; }
        public List<RelatedTopic> RelatedTopics { get; set; }
        public List<ResultItem> Results { get; set; }
        public string Type { get; set; }
        public Meta Meta { get; set; }
    }

    public class Infobox
    {
        public List<InfoboxItem> Content { get; set; }
        public List<InfoboxMeta> Meta { get; set; }
    }

    public class InfoboxItem
    {
        public string Data_Type { get; set; }
        public string Label { get; set; }
        public object Value { get; set; }
        public string Wiki_Order { get; set; } // can be both int and string
    }

    public class InfoboxMeta
    {
        public string Data_Type { get; set; }
        public string Label { get; set; }
        public string Value { get; set; }
    }

    public class RelatedTopic
    {
        public string FirstURL { get; set; }
        public Icon Icon { get; set; }
        public string Result { get; set; }
        public string Text { get; set; }
    }

    public class Icon
    {
        public string Height { get; set; }
        public string URL { get; set; }
        public string Width { get; set; }
    }

    public class ResultItem
    {
        public string FirstURL { get; set; }
        public Icon Icon { get; set; }
        public string Result { get; set; }
        public string Text { get; set; }
    }

    public class Meta
    {
        public string Description { get; set; }
        public string Dev_Milestone { get; set; }
        public List<Developer> Developer { get; set; }
        public string Example_Query { get; set; }
        public string Id { get; set; }
        public bool? Is_Stackexchange { get; set; }
        public string Js_Callback_Name { get; set; }
        public string Name { get; set; }
        public string Perl_Module { get; set; }
        public string Production_State { get; set; }
        public string Repo { get; set; }
        public string Signal_From { get; set; }
        public string Src_Domain { get; set; }
        public int Src_Id { get; set; }
        public string Src_Name { get; set; }
        public SrcOptions Src_Options { get; set; }
        public string Status { get; set; }
        public string Tab { get; set; }
        public List<string> Topic { get; set; }
        public int Unsafe { get; set; }
        public Maintainer Maintainer { get; set; }
    }

    public class Developer
    {
        public string Name { get; set; }
        public string Type { get; set; }
        public string Url { get; set; }
    }

    public class Maintainer
    {
        public string Github { get; set; }
    }

    public class SrcOptions
    {
        public string Directory { get; set; }
        public int Is_Fanon { get; set; }
        public int Is_Mediawiki { get; set; }
        public int Is_Wikipedia { get; set; }
        public string Language { get; set; }
        public string Min_Abstract_Length { get; set; }
        public int Skip_Abstract { get; set; }
        public int Skip_Abstract_Paren { get; set; }
        public string Skip_End { get; set; }
        public int Skip_Icon { get; set; }
        public int Skip_Image_Name { get; set; }
        public string Skip_Qr { get; set; }
        public string Source_Skip { get; set; }
        public string Src_Info { get; set; }
    }

}
