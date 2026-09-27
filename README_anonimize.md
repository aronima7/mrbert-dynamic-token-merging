  bash prepare_anonymous_repo.sh
  cd anonymous-submission
  git remote add origin git@github.com:<new-anonymous-account>/mrbert-anonymous.git
  git push -u origin main

---
=== Done ===
Anonymous repo created at: ./anonymous-submission

Next steps:
  1. Review the output: cd ./anonymous-submission && grep -r 'aronima\|stanford\|hivamoh' .
  2. Push to a fresh anonymous GitHub repo:
     cd ./anonymous-submission && git remote add origin <anonymous-repo-url> && git push -u origin main
  3. Submit via https://anonymous.4open.science using that repo URL

---