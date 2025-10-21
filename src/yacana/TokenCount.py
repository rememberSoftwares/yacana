class HuggingFaceDetails:
    def __init__(self, repo_name: str, token: str | None = None):
        self.repo_name = repo_name
        self.token = token
