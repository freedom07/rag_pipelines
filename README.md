## Result
The answer correctness score is assigned as an integer value between 1 and 5 by gpt-3.5-turbo. After scoring the entire evaluation dataset, min-max normalization was applied to scale the scores between 0 and 1.

| RAG Flow                                | Answer correctness score |
|-----------------------------------------|--------|
| baseline                                | 0.658  |
| baseline + hyde                         | 0.676  |
| baseline + query decompose              | 0.397  |
| baseline + hyde + bm25                  | 0.750  |
| baseline + hyde + hybrid                | 0.768  |
| baseline + hyde + colbert rerank        | 0.787  |
| baseline + hyde + hybrid + cohere rerank | 0.820  |
