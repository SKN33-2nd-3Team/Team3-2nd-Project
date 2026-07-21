# Git·GitHub 최초 설정 가이드

이 문서는 `SKN33-2nd-3Team/Team3-2nd-Project`에 처음 참여하는 팀원을 위한 설정 절차입니다.

## 1. 준비 사항

- Git 설치: `git --version`
- GitHub 계정 준비
- 저장소 Collaborator 또는 Organization 권한 확인
- HTTPS 사용 시 Git Credential Manager 또는 GitHub 인증 준비

## 2. 사용자 정보 설정

개인 PC 전체에 적용하려면 `--global`, 이 저장소에만 적용하려면 `--local`을 사용합니다.

```bash
git config --global user.name "GitHub 표시 이름"
git config --global user.email "GitHub에 등록한 이메일"

git config --global init.defaultBranch main
git config --global core.autocrlf true
```

확인:

```bash
git config --global --list
```

공용 PC에서는 개인정보가 남지 않도록 `--global` 대신 저장소 안에서 `--local` 사용을 권장합니다.

```bash
git config --local user.name "GitHub 표시 이름"
git config --local user.email "GitHub에 등록한 이메일"
```

## 3. 저장소 Clone

```bash
git clone https://github.com/SKN33-2nd-3Team/Team3-2nd-Project.git
cd Team3-2nd-Project
git remote -v
```

정상 remote:

```text
origin  https://github.com/SKN33-2nd-3Team/Team3-2nd-Project.git (fetch)
origin  https://github.com/SKN33-2nd-3Team/Team3-2nd-Project.git (push)
```

이미 Clone했는데 주소가 다를 때만 수정합니다.

```bash
git remote set-url origin https://github.com/SKN33-2nd-3Team/Team3-2nd-Project.git
git remote -v
```

## 4. Python 환경 설정

PowerShell 기준:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

가상환경과 비밀정보는 커밋하지 않습니다.

## 5. 작업 시작

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/작업이름
```

예시:

```bash
git switch -c feature/soft-voting
git switch -c fix/feature-order
git switch -c docs/update-readme
```

## 6. 변경 확인과 Commit

```bash
git status
git diff

# 필요한 파일만 선택
git add src/model.py tests/test_model.py

# 실제 커밋 대상 최종 확인
git diff --staged
git commit -m "feat: Soft Voting 앙상블 추가"
```

`git add .`를 사용할 때는 비밀정보, 데이터, 모델, 임시 파일이 포함되지 않았는지 반드시 먼저 확인합니다.

## 7. Push와 Pull Request

```bash
git push -u origin feature/soft-voting
```

GitHub에서 표시되는 **Compare & pull request**를 선택하고 PR 템플릿을 작성합니다.

PR 병합 전 확인:

- 변경 목적이 하나인가?
- 로컬 검증을 통과했는가?
- CI가 통과했는가?
- 비밀정보와 불필요한 파일이 없는가?
- 모델 결과와 README 수치가 일치하는가?
- 최소 1명이 리뷰했는가?

## 8. PR 병합 후 정리

```bash
git switch main
git pull --ff-only origin main
git branch -d feature/soft-voting
```

GitHub에서 원격 브랜치 자동 삭제 옵션을 켰다면 별도 원격 삭제는 필요하지 않습니다.

## 9. 다른 팀원의 변경 가져오기

현재 변경이 없을 때:

```bash
git switch main
git pull --ff-only origin main
```

작업 중인 변경이 있다면 먼저 커밋하거나 별도 브랜치에 보관한 뒤 동기화합니다. 다른 팀원의 파일을 덮어쓰기 위해 `reset --hard`를 사용하지 않습니다.

## 10. 자주 만나는 문제

### Push 권한이 없음

- Organization 또는 저장소 Collaborator 초대를 수락했는지 확인합니다.
- 올바른 GitHub 계정으로 인증했는지 확인합니다.
- `git remote -v`가 현재 저장소를 가리키는지 확인합니다.

### 이메일이 잘못 기록됨

```bash
git config --local user.email "GitHub에 등록한 이메일"
```

### PowerShell에서 가상환경 실행이 차단됨

현재 터미널 세션에서만 다음을 적용한 후 다시 활성화합니다.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 변경 파일이 너무 많음

```bash
git status
git diff
```

가상환경, cache, 임시 결과가 포함됐다면 `.gitignore`를 먼저 확인합니다.

## 11. GitHub 저장소 관리자 설정

아래 설정은 저장소 관리자 권한이 있는 팀원이 GitHub 웹에서 수행합니다.

### Settings → General → Pull Requests

- `Allow squash merging` 활성화
- 불필요하면 Merge commit·Rebase merge 비활성화
- `Automatically delete head branches` 활성화

### Settings → Rules → Rulesets

`main` 대상 Ruleset을 만들고 다음을 권장합니다.

- Pull Request를 통해서만 변경
- 승인 최소 1명
- 리뷰 대화 해결 필수
- 상태 검사 `quality` 통과 필수
- Force push 차단
- Branch 삭제 차단

### Settings → Actions → General

- Workflow permissions는 기본적으로 `Read repository contents` 사용
- 외부 Action은 검증된 공식 Action을 우선 사용

### Issues와 Project

- Issues 활성화
- 필요하면 `Backlog → Ready → In Progress → Review → Done` 보드 구성
- 모델 실험은 Issue에 가설·고정 조건·변경 조건·결과를 남김

## 12. 관련 문서

- [협업 규칙](../CONTRIBUTING.md)
- [Pull Request 템플릿](../.github/pull_request_template.md)
- [프로젝트 README](../README.md)
