import sys; sys.path.insert(0, ".")
from app.chain import answer_question

result = answer_question("What is machine learning?")
print("Question:", result["question"])
print("Answer:", result["answer"])
print("Sources:")
for s in result["sources"]:
    print(f"  - {s['source']}, page {s['page']}")