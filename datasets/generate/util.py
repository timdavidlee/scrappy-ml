import git


def get_git_root():
    repo = git.Repo('.', search_parent_directories=True)
    return repo.working_tree_dir
