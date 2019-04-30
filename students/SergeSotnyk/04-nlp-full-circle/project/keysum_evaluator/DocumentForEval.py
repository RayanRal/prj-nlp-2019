from typing import List


class DocumentForEval:
    """
    Container for information for evaluation metrics of automated summarising
    and keywords extraction quality.
    """

    def __init__(self,
                 ref_keywords: List[str] = None,
                 keywords: List[str] = None,
                 ref_summary: List[str] = None,
                 summary: List[str] = None,
                 lang: str = "en"
                 ):
        """
        Constructor

        :param ref_keywords: referenced ("ideal", manually created) keywords
        :param keywords: automatically created keywords
        :param ref_summary: sentences of manually created summary
        :param summary: sentences of automatically created summary
        :param lang: document language - some languages demand additional language-depended processing
        """
        self.ref_keywords = [] if ref_keywords is None else ref_keywords
        self.keywords = [] if keywords is None else keywords
        self.ref_summary = [] if ref_summary is None else ref_summary
        self.summary = [] if summary is None else summary
        self.lang = lang

