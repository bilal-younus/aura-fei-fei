"""Run all required standalone parts and preserve their actual terminal output."""
import contextlib,io,json,platform,sys,time
from common import ROOT,EVIDENCE,setup,load_part,save_json

def main():
    setup();start=time.time()
    for filename,entry in [('01_DPO_Loss_Core.py','gradient_demo'),('02_Implicit_Reward_Evaluator.py','run'),('03_Length_Bias_and_SimPO.py','run'),('04_Activation_Steering_and_Audit.py','run')]:
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            value=getattr(load_part(filename),entry)()
            if entry=='gradient_demo':print(json.dumps(value,indent=2))
        output=stream.getvalue();(EVIDENCE/(filename[:2]+'_output.txt')).write_text(output)
        print('\n'+filename+'\n'+output,flush=True)
    save_json('execution.json',dict(python=sys.version,platform=platform.platform(),elapsed_seconds=time.time()-start,command='python run_all.py',completed_parts=4))
if __name__=='__main__':main()
