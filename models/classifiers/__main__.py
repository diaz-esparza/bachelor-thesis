## FOR TESTING PURPOSES

def main():
    from os.path import join as jo
    import os.path as osp
    import sys
    sys.path.append(jo(osp.dirname(__file__),".."))
    
    from classifiers import train_classifier,evaluate_ckpt
    for seed in range(4):
        train_classifier(seed,include_fake_data=True)
    for seed in range(4):
        train_classifier(seed,include_fake_data=False)
    #evaluate_ckpt("/home/sergio/Documents/AM/internal/storage/original/ckpt/inception_custom_synth_40.ckpt")
    

if __name__=="__main__": main()
