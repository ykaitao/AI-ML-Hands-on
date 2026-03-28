#!/bin/sh
git fetch && git rebase origin/master && git push -f
# List local branches for verification purposes
git branch
