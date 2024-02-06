This module allows you to scan DM codes using a webcam  
The libdmtx library is used in the py wrapper  
To work, you need to install the related modules  
pip install pylibdmtx  
pip install opencv-python  
pip install paperclip  

Distributed without support and without any guarantees. Available  
commercial and non-commercial use and generally any  
use is subject to the preservation of authorship.  

Usage examples:  
ascortDmScanner // open the program with default parameters and show the scan mode on the screen  
ascortDmScanner --camID=1 --clipboard=True // Use camera number 1 and copy to clipboard as soon as UIN is found  
ascortDmScanner --readFromFile='qr.jpg' --resultFile='result.txt ' // read the file and save the WIN to a file result.txt "'  

Myagkov Anton @ wow1c.com
a@wow1c.com, 2023  
