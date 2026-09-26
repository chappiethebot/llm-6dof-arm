import time
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
c = OpenAI()
ok, errs = 0, []
for i in range(30):
    try:
        c.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "Reply OK"}], max_tokens=2)
        ok += 1
    except Exception as e:
        errs.append(type(e).__name__ + " | cause: " + repr(e.__cause__)[:200])
    time.sleep(0.5)
print("ok", ok, "of 30")
for e in errs[:5]:
    print(e)
