import os, sys
import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
filt = sys.argv[1].lower() if len(sys.argv) > 1 else ""
names = [m.name.removeprefix("models/") for m in genai.list_models()
         if "generateContent" in getattr(m, "supported_generation_methods", [])]
for n in sorted(names):
    if filt in n.lower():
        print(n)
print()
print("'gemini-2.5-flash' available:", "gemini-2.5-flash" in names)
