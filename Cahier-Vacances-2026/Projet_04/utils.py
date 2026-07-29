def print_chat(question, answer, sources=None):
    print(f"Client   : {question}")
    if sources is not None:
        titles = ", ".join(
            f"{row.Title} ({row.Score:.2f})"
            for row in sources.itertuples(index=False)
        )
        print(f"Sources  : {titles}")
    print(f"Assistant: {answer}")
