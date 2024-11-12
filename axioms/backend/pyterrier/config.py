RETRIEVAL_SCORE_APPLICATION_PROPERTIES = {
    "querying.processes": ",".join([
        "terrierql:TerrierQLParser",
        "parsecontrols:TerrierQLToControls",
        "parseql:TerrierQLToMatchingQueryTerms",
        "applypipeline:ApplyTermPipeline",
        "context_wmodel:org.terrier.python.WmodelFromContextProcess",
        "localmatching:LocalManager$ApplyLocalMatching",
        "filters:LocalManager$PostFilterProcess"
    ]),
    "querying.postfilters": "decorate:SimpleDecorate",
    "querying.default.controls": ",".join([
        "parsecontrols:on",
        "parseql:on",
        "applypipeline:on",
        "terrierql:on",
        "localmatching:on",
        "filters:on",
        "decorate:on"
    ]),
}
