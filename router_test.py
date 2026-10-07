from src.model_router import choose_model


questions = [
    "What is an AutoEncoder?",
    "Compare AutoEncoders and PCA.",
    "Explain why autoencoders are useful.",
    "What is the purpose of an encoder?"
]


for question in questions:

    model = choose_model(question)

    print("\nQuestion:", question)
    print("Selected model:", model)