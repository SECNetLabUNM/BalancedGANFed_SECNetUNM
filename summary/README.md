
## new playbooks and their commands:
```
$ ansible-playbook -i hosts-current.ini Copilot-Manage-repositories-and-find-sqlite-files.yml  --extra-vars "@vars/paths-lobogit-graphganfeddrugbank.yml"  --tags git_commit_date |grep -e "item\":" -e  "stdout\""  -e localhost -e m4mini -e server1070 -e server1080 | grep -B 1 "2025-03"
$ ansible-playbook -i hosts-current.ini Copilot-Manage-repositories-and-find-sqlite-files.yml  --extra-vars "@vars/paths.yml"  --tags git_commit_date  |grep -e "item\":" -e  "stdout\""  -e localhost -e m4mini -e server1070 -e server1080
$ ansible-playbook -i hosts-current.ini Copilot-Manage-repositories-and-find-sqlite-files.yml   --tags  git_pull
```
* copilot explanations
  * Exactly! When you use --extra-vars, the variables specified there will take precedence over those defined in the vars_files directive within the playbook. They won't concatenate; instead, any overlapping variables will be overridden by the values from --extra-vars.
If you need to combine variables from multiple files, you would need to manually merge them into a single file or handle the merging logic within your playbook.
