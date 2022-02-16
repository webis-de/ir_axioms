from ir_axioms.model import base, context

# Re-export sub-modules.
Query = base.Query
Document = base.Document
TextDocument = base.TextDocument
RankedDocument = base.RankedDocument
RankedTextDocument = base.RankedTextDocument
JudgedRankedDocument = base.JudgedRankedDocument
JudgedRankedTextDocument = base.JudgedRankedTextDocument

IndexContext = context.IndexContext
