
class BaseChunker:
    def __init__(self, chunk_size: int):
        self.content: str = ""
        self.chunk_size = chunk_size

    def chunk(self, data: str) -> list[str]:
        raise NotImplementedError("Subclasses should implement this method.")


class TextChunker(BaseChunker):
    def chunk(self, data: str) -> list[str]:
        """Strategy: cut at the last return character before limit
            or cut the last sentence (finished by .?!) before limit
            or cut at the last space if no return or sentence is found."""
        self.content = data
        chunks: list[str] = []
        while len(data) > self.chunk_size:
            # Find the last newline character before limit
            last_newline = data.rfind('\n', 0, self.chunk_size)
            if last_newline != -1:
                chunks.append(data[:last_newline])
                data = data[last_newline + 1:]
                continue

            # Find the last sentence punctuation before  limit
            punc: list[str] = ['.', '?', '!', ':', ';']
            last_sentence_end = max(
                data.rfind(punct, 0, self.chunk_size) for punct in punc)
            if last_sentence_end != -1:
                chunks.append(data[:last_sentence_end + 1])
                data = data[last_sentence_end + 1:]
                continue

            # Find the last space before limit
            last_space = data.rfind(' ', 0, self.chunk_size)
            if last_space != -1:
                chunks.append(data[:last_space])
                data = data[last_space + 1:]
                continue

            # Wtf I am just cutting gibberish
            chunks.append(data[:self.chunk_size])
            data = data[self.chunk_size:]
        return chunks + [data] if data else chunks


class PythonChunker(BaseChunker):
    """"""
    def chunk(self, data: str) -> list[str]:
        self.content = data
        chunks: list[str] = []
        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks


class ChunkerFactory:
    @staticmethod
    def get_chunker(file_extension: str, chunk_size: int) -> BaseChunker:
        if file_extension == ".py":
            return PythonChunker(chunk_size)
        else:
            return TextChunker(chunk_size)


if __name__ == "__main__":
    # Example usage
    text_chunker = TextChunker(chunk_size=45)
    text_data = "This is a sample text. It will be chunked into smaller \
    pieces based on the specified chunk size."
    text_chunks = text_chunker.chunk(text_data)
    print("Text Chunks:", text_chunks)

    python_chunker = PythonChunker(chunk_size=50)
    python_data = "def example_function():\n    print('This is a sample \
        Python function.')\n\nexample_function()"
    python_chunks = python_chunker.chunk(python_data)
    print("Python Chunks:", python_chunks)
