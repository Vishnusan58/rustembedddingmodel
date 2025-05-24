# Uploading to GitHub

This document provides step-by-step instructions for uploading this project to GitHub.

## Prerequisites

1. [Create a GitHub account](https://github.com/join) if you don't already have one
2. [Install Git](https://git-scm.com/downloads) on your local machine
3. Configure Git with your username and email:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

## Steps to Upload to GitHub

### 1. Create a New Repository on GitHub

1. Log in to your GitHub account
2. Click on the "+" icon in the top-right corner and select "New repository"
3. Enter "embedddingmodel" as the repository name
4. Add a description (optional): "A Rust application that generates text embeddings using FastText"
5. Choose whether the repository should be public or private
6. Do NOT initialize the repository with a README, .gitignore, or license (we already have these files)
7. Click "Create repository"

### 2. Initialize Git in Your Local Project

Open a command prompt or terminal and navigate to your project directory:

```bash
cd C:\Users\vishn\RustroverProjects\embedddingmodel
```

Initialize a Git repository:

```bash
git init
```

### 3. Add Your Files to Git

Add all files to the staging area:

```bash
git add .
```

Commit the files:

```bash
git commit -m "Initial commit: Text Embedding Model"
```

### 4. Connect Your Local Repository to GitHub

Link your local repository to the GitHub repository you created:

```bash
git remote add origin https://github.com/yourusername/embedddingmodel.git
```

Replace `yourusername` with your actual GitHub username.

### 5. Push Your Code to GitHub

Push your code to the main branch on GitHub:

```bash
git push -u origin main
```

If your default branch is named "master" instead of "main", use:

```bash
git push -u origin master
```

### 6. Verify the Upload

1. Go to your GitHub repository page: `https://github.com/yourusername/embedddingmodel`
2. Refresh the page if necessary
3. You should see all your files uploaded to GitHub

## Additional Tips

### .gitignore

Consider adding or updating a `.gitignore` file to exclude unnecessary files from your repository:

```bash
# Create or edit .gitignore
echo "/target/" >> .gitignore
echo "**/*.rs.bk" >> .gitignore
echo "Cargo.lock" >> .gitignore
git add .gitignore
git commit -m "Add .gitignore file"
git push
```

### GitHub Pages

If you want to create a website for your project:

1. Go to your repository on GitHub
2. Click on "Settings"
3. Scroll down to the "GitHub Pages" section
4. Select the branch you want to use (usually "main")
5. Click "Save"

Your project website will be available at: `https://yourusername.github.io/embedddingmodel/`

## Troubleshooting

- **Authentication issues**: If you're having trouble authenticating, consider using a personal access token or SSH key
- **Push rejected**: If your push is rejected, try pulling first with `git pull --rebase origin main`
- **Large files**: If you have large files that exceed GitHub's file size limit, consider using Git LFS