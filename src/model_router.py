def choose_model(question):

    # Convert question to lowercase
    question = question.lower()

    # Words that usually indicate a more complex question
    complex_words = [
        "compare",
        "difference",
        "explain why",
        "analyze",
        "analyse",
        "evaluate",
        "advantages and disadvantages",
        "pros and cons",
        "compare and contrast"
    ]

    # Check for complex questions
    for word in complex_words:

        if word in question:
            return "openai/gpt-oss-120b:groq"

    # Otherwise use the cheaper model
    return "openai/gpt-oss-20b:groq"