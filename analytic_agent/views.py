from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['POST'])
def ask_retail_rag_ui(request):
    from analytics_bot.src.pipeline import ask_retail_rag_ui
    question = request.data.get("question", "")
    out = ask_retail_rag_ui(question)
    return Response(out)


{    "question": "What is the best selling product in the last month?"}