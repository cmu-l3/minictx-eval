import threading
import pexpect
import json
import os
import time
import tempfile
import re

class InteractiveThread(threading.Thread):
    def __init__(self, session_id, repl_path, lean_env_path, initial_context = None, timeout=600):
        super().__init__()
        self.session_id = session_id
        self.repl_path = os.path.abspath(repl_path)
        self.lean_env_path = os.path.abspath(lean_env_path)
        self.context = initial_context
        self.session = None
        
        self.cmd_response_condition = threading.Event()
        self.cmd_query_condition = threading.Event()
        self.init_complete = threading.Event()
        self.response = None

        self.stop_flag = False
        self.timer = threading.Timer(timeout, self.stop) 

    def initialize_check(self):
        try:
            if self.context == None:
                initialize_check = {"cmd": "def init_check : Nat := 42"}
                self.send_cmd(initialize_check)
            self.session.expect('"env": 0}\r\n\r\n', timeout=60)  #If context contains sorries, it will have more keys other than env
            self.init_complete.set()
            # print(f"Session {self.session_id}: Successfully start repl and waiting for input.")
        except:
            self.init_complete.set()
            print(f"Session {self.session_id}: fail to initialize lean repl")
            print(self.context)
            print(self.session.before)
            self.stop()
            # self.join()

    def send_cmd(self, cmd):
        cmd_str = json.dumps(cmd, ensure_ascii=False) 
        self.session.sendline(cmd_str + '\n')

    def submit_and_receive(self, cmd):
        if self.stop_flag: return None

        self.init_complete.wait()
        
        self.send_cmd(cmd)
        
        self.cmd_query_condition.set() 

        self.cmd_response_condition.wait()  # wait for the response
        self.cmd_response_condition.clear()
        if self.response:
            output = self.response
            self.response = None
            return output  
        return None

    def process_responses(self):
        while not self.stop_flag:
            self.cmd_query_condition.wait() #wait for input 
            self.cmd_query_condition.clear()

            if self.stop_flag:  #terminate session
                break

            try:
                self.session.expect('\r\n\r\n', timeout=10) #filter out input, pexpect print the input twice for unknown reason
                self.session.expect(['\r\n\r\n', pexpect.EOF], timeout=10)
                output = self.session.before.strip()
                output_dict = json.loads(output)
                # print("Received:", output_dict)

                self.response = output_dict
                self.cmd_response_condition.set()  

            except pexpect.TIMEOUT:
                print("Output timeout")  
                break # Terminate session
            except pexpect.EOF:
                print("Session ended unexpectedly.")
                break
            except json.JSONDecodeError as e:
                print(output)
                break
    
    def remove_last_comment(self):
        pattern = r'/--[^/]*?-/(\n*)$'
        self.context = re.sub(pattern, '', self.context, flags=re.DOTALL)

    def run(self):
        self.timer.start() 
        try:
            self.session = pexpect.spawn('bash', encoding='utf-8', cwd=self.lean_env_path)
            if self.context != None:
                self.remove_last_comment()
                with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
                    json.dump({"cmd": self.context}, temp, ensure_ascii=False)
                    temp.write("\n\n")
                    temp.flush()
                command = f'lake env {self.repl_path}/.lake/build/bin/repl < <(cat {temp.name} -)'
            else:
                command = f'lake env {self.repl_path}/.lake/build/bin/repl'
            
            self.session.sendline(command)
            self.initialize_check()
            self.process_responses()  # Continuously process responses
            self.stop()
    
        except Exception as e:
            print(f"Session {self.session_id}: An error occurred: {e}")
            self.stop()

    def stop(self):
        self.stop_flag = True
        self.init_complete.set()
        self.cmd_query_condition.set() 
        self.cmd_response_condition.set()  
        self.timer.cancel()


if __name__ == "__main__":
    repl_path = './repl'
    lean_env_path = './PrimeNumberTheoremAnd'
    
    test = "import Mathlib.Analysis.Complex.CauchyIntegral\nimport Mathlib.Analysis.Complex.Convex\n\nopen Complex Set Topology\n\nopen scoped Interval\n\nvariable {z w : ℂ} {c : ℝ}\n\n/-%%\nThis files gathers definitions and basic properties about rectangles.\n%%-/\n\n/-%%\n\\begin{definition}\\label{Rectangle}\\lean{Rectangle}\\leanok\nA Rectangle has corners $z$ and $w \\in \\C$.\n\\end{definition}\n%%-/\n/- A `Rectangle` has corners `z` and `w`. -/\ndef Rectangle (z w : ℂ) : Set ℂ := [[z.re, w.re]] ×ℂ [[z.im, w.im]]\n\nnamespace Rectangle\n\n"

    thread = InteractiveThread(1, repl_path, lean_env_path, initial_context = test)
    thread.start()

    # output = thread.submit_and_receive(cmd)
    # print(output)

    test = '''lemma symm : Rectangle z w = Rectangle w z := by sorry'''
    
    output = thread.submit_and_receive({"cmd": test, "env": 0})
    print(output)

    # output = thread.submit_and_receive({"cmd": "theorem amc12_2000_p11 (a b : ℝ) (h₀ : a ≠ 0 ∧ b ≠ 0) (h₁ : a * b = a - b) :\n    a / b + b / a - a * b = 2 := by sorry", "env": 1})
    # print(output)
    # output = thread.submit_and_receive({"tactic": "field_simp [h₀.1, h₀.2]", "proofState": 0})
    # print(output)
    # thread.submit_and_receive({"tactic": "simp only [h₁, mul_comm, mul_sub]", "proofState": 1})
    # thread.submit_and_receive({"tactic": "ring", "proofState": 2})
    # print("Output:", output)
    thread.stop()
    thread.join()


