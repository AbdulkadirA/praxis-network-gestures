
import teem
import ctypes
from numpy import *
from math import sqrt
from math import exp
import sys
import os


def nrrdToCType( ntype ):
    """For a gived nrrd type, return a typle with the corresponding ctypes type, array typecode and numpy type"""
    typeTable = {
        teem.nrrdTypeChar : ctypes.c_byte,
        teem.nrrdTypeUChar : ctypes.c_ubyte,
        teem.nrrdTypeShort : ctypes.c_short,
        teem.nrrdTypeUShort : ctypes.c_ushort,
        teem.nrrdTypeInt : ctypes.c_int,
        teem.nrrdTypeUInt : ctypes.c_uint,
        teem.nrrdTypeLLong : ctypes.c_longlong,
        teem.nrrdTypeULLong : ctypes.c_ulonglong,
        teem.nrrdTypeFloat : ctypes.c_float,
        teem.nrrdTypeDouble : ctypes.c_double
    }
    return typeTable[ntype] 



def computeconn(conn1, label1, conn2, label2, labelmapfile):


  n1 = teem.nrrdNew()
  n2 = teem.nrrdNew()
  mask = teem.nrrdNew()

  teem.nrrdLoad(mask,labelmapfile,None) 
  teem.nrrdLoad(n1, conn1, None) 
  teem.nrrdLoad(n2, conn2, None)
 
  maskdata = ctypes.cast(mask.contents.data, ctypes.POINTER(nrrdToCType(mask.contents.type)))
  n1data = ctypes.cast(n1.contents.data, ctypes.POINTER(nrrdToCType(n1.contents.type)))
  n2data = ctypes.cast(n2.contents.data, ctypes.POINTER(nrrdToCType(n2.contents.type)))

  su = mask.contents.axis[0].size
  sv = mask.contents.axis[1].size
  sw = mask.contents.axis[2].size

  sum1 = 0
  sum2 = 0
  ct1 = 0
  ct2 = 0

  for ui in range(su):
    uu = ui
    for vi in range(sv):
      vv = vi
      for wi in range(sw):
        ww = wi

        index = (sv*wi + vi)*su + ui

        if(maskdata[index] == label2):
          sum1 = sum1 + n1data[index]
          ct1 = ct1 + 1
        if(maskdata[index] == label1):
          sum2 = sum2 + n2data[index]
          ct2 = ct2 + 1
 

  teem.nrrdNuke(n2)
  teem.nrrdNuke(n1)
  teem.nrrdNuke(mask)

  c =  ((sum1/ct1) + (sum2/ct2))/2
  cexp = exp(-0.1*c); #default: -0.1. 
  #print "Done."

  return cexp 
  
############
#   main   #
############


DATAPATH='/data/pnl/Petra/Peter_Finsler/output/'
OUTPUTDIR = DATAPATH + 'conn_matrices/'

case=sys.argv[1]
#echo $case

labelmap = '/data/pnl/Petra/Registration/' + case + 'registration/wmparc-in-bse.nrrd'

#if not os.path.exists(labelmap + '.nrrd'):
#   os.system('ConvertBetweenFileFormats ' + labelmap + '.nii.gz ' + labelmap + '.nrrd')
#labelmap = labelmap + '.nrrd'

with open('/data/pnl/Petra/Peter_Finsler/scripts/allcombs.txt') as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]
f.close()

connmat = zeros([68,68])

with open('/data/pnl/Petra/Peter_Finsler/scripts/uniquelabels.txt') as f:
    line = f.readlines()
f.close()

labels = line[0].strip()
labels_num =  [float(x) for x in labels.split(' ')]



for i in range(len(lines)):
    s = lines[i]
    l = s.split()
    lf =  [float(x) for x in l]
    #print lf
    #print labels_num
    conn1 = DATAPATH + case + '/' + case +'_connMap_' + str(int(float(l[0]))) + '.nhdr'
    conn2 = DATAPATH + case + '/' + case +'_connMap_' + str(int(float(l[1]))) + '.nhdr' 

    print conn1
    print conn2

    if (os.path.exists(conn1) and os.path.exists(conn2)):
       c = computeconn(conn1,lf[0],conn2,lf[1],labelmap)
       ii = labels_num.index(lf[0])
       jj = labels_num.index(lf[1])

       connmat[ii,jj] = c


connmat = connmat + connmat.transpose()

#outname = outputdir + f1[0:11] + '_connmat_meanFA_tensor1.txt'
outname = OUTPUTDIR + case + '_connmat_Finsler.txt'

#numpy.savetxt(outname, connmat, fmt='%.6f', delimiter=' ')
savetxt(outname, connmat, fmt='%f', delimiter=' ' )       
