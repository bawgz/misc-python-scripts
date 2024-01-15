from huggingface_hub import HfApi
import sys


def main(path_to_file, path_in_repo, repo_id, repo_type):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=path_to_file,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
    )

if __name__ == "__main__":
    path_to_file = sys.argv[1]
    path_in_repo = sys.argv[2]
    repo_id = sys.argv[3]
    repo_type = sys.argv[4]
    main(path_to_file, path_in_repo, repo_id, repo_type)