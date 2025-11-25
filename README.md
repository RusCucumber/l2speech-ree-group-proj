# Research Project Template

This repository is a research project template using R and Python.

# How to Start?

## 1. rye
```{bash}
rye init .
rye sync
```

## 2. Ruff
1. Run following scripts

```{bash}
rye add --dev ruff
rye sync
```

2. Modify .vscode/settings.json
```{json}
{
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
    "python.analysis.extraPaths": [
        "./src/PROJECT_NAME"
    ]
}
```

3. Commit change

## 3. R
1. Set renv
```{R}
renv::init()
```

2. Install pacman
```{r}
install.packages("pacman")
```

3. Snapshot installed packages
```{r}
renv::snapshot()
```
