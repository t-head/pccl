import re
import subprocess


AI_AUTHOR_NAME_PATTERN = re.compile(
    r'\b(?:claude code|copilot|codex|cursor|gemini cli|gemini code assist|qoder|devin ai|ai assistant)\b',
    re.IGNORECASE)
AI_AUTHOR_EMAILS = {
    'noreply@anthropic.com', 'noreply@openai.com', 'cursoragent@cursor.com',
}


def validate_attribution(message):
    # Squash bodies can contain only trailers, without the title Git's parser expects.
    trailers = subprocess.check_output(
        ['git', 'interpret-trailers', '--parse', '--no-divider'], input='Attribution check\n\n' + message,
        text=True).splitlines()
    attribution = []
    for line in trailers:
        trailer = re.fullmatch(r'(?:Co-authored-by|Signed-off-by):\s*(.*)', line, re.IGNORECASE)
        if not trailer:
            continue
        identity = re.fullmatch(r'(.*?)\s*<([^<>\s]+)>', trailer[1])
        name, email = identity.groups() if identity else (trailer[1], '')
        if AI_AUTHOR_NAME_PATTERN.search(' '.join(name.split())) or email.casefold() in AI_AUTHOR_EMAILS:
            raise ValueError('AI tool attribution is not allowed in Co-authored-by or Signed-off-by; '
                             'remove it from the commit message and preserve human co-authors and DCO sign-offs')
        attribution.append(line)
    return attribution
