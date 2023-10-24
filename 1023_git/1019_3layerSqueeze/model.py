import numpy as np

def model(pars):
    Ninp = pars['Nin'] # number of neurons in the input layer
    Nin = int(Ninp/2)
    NE1 = pars['NE1'] # number of neurons in E1 layer
    NB = pars['NB']
    a = pars['a']
    b = pars['b']
    c = pars['c']
    d = pars['d']

    taup = np.random.normal(10,2,Nin)
    taun = np.random.normal(10,2,Nin)
    vppeak = np.random.normal(10,2,Nin)
    vnpeak = np.random.normal(10,2,Nin)

    T = 2000 #Total time in ms
    dt = 0.1 #Integration time step in ms 
    nt = int(T/dt) #Time steps
    tvec = np.linspace(0,T,nt)
    tref = 5 # refractory period in milliseconds
    tlastp = np.zeros(Nin)
    tlastn = np.zeros(Nin)
    tlastE1 = np.zeros(NE1)
    tlastB = np.zeros(NB)
    # voltage arrays
    vpos = np.zeros((Nin,nt))
    vneg = np.zeros((Nin,nt))
    vE1 = np.zeros((NE1,nt))
    vB = np.zeros((NB,nt))
    # adaptation arrays
    upos = np.zeros((Nin,nt))
    uneg = np.zeros((Nin,nt))
    uE1 = np.zeros((NE1,nt))
    uB = np.zeros((NB,nt))
    # post synaptic currents 
    IPSCE1 = np.zeros(NE1)
    IPSCB = np.zeros(NB)
    # JD's
    JDpos = 0*IPSCE1
    JDneg = 0*IPSCE1
    JDE1 = 0*IPSCB
    # h's
    hE1 = np.zeros(NE1)
    hB = np.zeros(NB)
    # r's
    rE1 = np.zeros(NE1)
    rB = np.zeros(NB)
    # hr's
    hrE1 = np.zeros(NE1)
    hrB = np.zeros(NB)
    ## Izhikevich Parameters
    C = 1  #capacitance
    vr = -60   #resting membrane 
    vpeak = 30  # peak voltage
    # vreset = -65 # reset voltage 
    tr = 2  #synaptic rise time 
    td = 20 #decay time
    vreset = c # reset voltage 
    p = 0.3 #connectivity sparsity 
    G = 5e2 #Gain on the static matrix with 1/sqrt(N) scaling weights. 
    BIAS = 0; #Bias current, note that the Rheobase is around 950
    # f1 = 0.001
    f1 = 0.002
    f2 = 0.005
    f3 = 0.010
    amp = 0.5
    sos = amp*np.sin(2*np.pi*f1*tvec) + amp*np.sin(2*np.pi*f2*tvec) - amp*np.sin(2*np.pi*f3*tvec)
    vresetQIF = -0.1 # reset voltage

    #-----Initialization---------------------------------------------
    vpos[:,0] = vresetQIF+(vppeak-vresetQIF)*np.random.rand(Nin)
    vneg[:,0] = vresetQIF+(vnpeak-vresetQIF)*np.random.rand(Nin)
    vE1[:,0] = vr+(vpeak-vr)*np.random.rand(NE1)
    vB[:,0] = vr+(vpeak-vr)*np.random.rand(NB)
    upos[:,0] = np.zeros(Nin)
    uneg[:,0] = np.zeros(Nin)
    uE1[:,0] = np.zeros(NE1)
    uB[:,0] = np.zeros(NB)

    #----weight matrices -------------------------------------------
    OMEGApos = G*(np.random.rand(NE1,Nin)<p)*(np.random.randn(NE1,Nin))/np.sqrt(Nin*p)
    OMEGAneg = G*(np.random.rand(NE1,Nin)<p)*(np.random.randn(NE1,Nin))/np.sqrt(Nin*p)
    OMEGAE1 = G*(np.random.rand(NB,NE1)<p)*(np.random.randn(NB,NE1))/np.sqrt(NE1*p)

    #----- arrays for spike times and spike indices in each layer---
    stpos = []
    sipos = []
    stneg = []
    sineg = []
    stE1 = []
    siE1 = []
    stB = []
    siB = []

    # integrate with forward Euler
    for n in range(nt-1):
        #--------------------- positively-tuned input layer ----------------------------------
        idxpos = np.argwhere(vpos[:,n] >= vppeak)
        dv = (dt*n>tlastp + tref)*(vpos[:,n]**2 + np.ones(Nin)*(sos[n] > 0)*(sos[n]))/taup
        vpos[:,n+1] = dv*dt + vpos[:,n]
        if len(idxpos) > 0:
            spike_times = np.ones(len(idxpos))*n*dt
            stpos.extend(spike_times)
            spike_indices = idxpos[:,0]
            sipos.extend(spike_indices)
            JDpos = np.sum(OMEGApos[:,idxpos],axis=1)[:,0]
            vpos[:,n+1] = vpos[:,n+1] + (vresetQIF - vpos[:,n+1])*(vpos[:,n] > vppeak)
        tlastp = tlastp + (dt*n-tlastp)*(vpos[:,n+1]>=vppeak) # used for refractory period

        #--------------------- negatively-tuned input layer ----------------------------------
        idxneg = np.argwhere(vneg[:,n] >= vnpeak)
        dv = (dt*n>tlastn + tref)*(vneg[:,n]**2 + np.ones(Nin)*(sos[n] < 0)*(-sos[n]))/taun
        vneg[:,n+1] = dv*dt + vneg[:,n]
        if len(idxneg) > 0:
            spike_times = np.ones(len(idxneg))*n*dt
            stneg.extend(spike_times)
            spike_indices = idxneg[:,0]
            sineg.extend(spike_indices)
            JDneg = np.sum(OMEGAneg[:,idxneg],axis=1)[:,0]
            vneg[:,n+1] = vneg[:,n+1] + (vresetQIF - vneg[:,n+1])*(vneg[:,n] > vnpeak)
        tlastn = tlastn + (dt*n-tlastn)*(vneg[:,n+1]>=vnpeak) # used for refractory period

        #--------------------- expansion 1 (E1) layer ----------------------------------
        IE1 = IPSCE1 #post-synaptic currents to E1
        dv = (dt*n>tlastE1 + tref)*(0.04*vE1[:,n]**2 + 5*vE1[:,n] + 140 - uE1[:,n] + IE1)/C
        du = (a*(b*(vE1[:,n]-vr)-uE1[:,n]))
        vE1[:,n+1] = vE1[:,n] + dv*dt
        uE1[:,n+1] = uE1[:,n] + du*dt
        idxE1 = np.argwhere(vE1[:,n+1]>=vpeak)
        if len(idxE1)>0:
            spike_times = np.ones(len(idxE1))*n*dt
            stE1.extend(spike_times)
            spike_indices = idxE1[:,0]
            siE1.extend(spike_indices)
            JDE1 = np.sum(OMEGAE1[:,idxE1],axis=1)[:,0]
        tlastE1 = tlastE1 + (dt*n-tlastE1)*(vE1[:,n+1]>=vpeak)
        IPSCE1 = IPSCE1*np.exp(-dt/tr) + hE1*dt
        hE1 = hE1*np.exp(-dt/td) + JDpos*(len(idxpos)>0)/(tr*td) + JDneg*(len(idxneg)>0)/(tr*td)
        uE1[:,n+1] = uE1[:,n+1] + d*(vE1[:,n+1]>=vpeak)
        vE1[:,n+1] = vE1[:,n+1] + (vreset-vE1[:,n+1])*(vE1[:,n+1]>=vpeak)

        #--------------------- bottleneck (B) layer ----------------------------------
        IB = IPSCB #post-synaptic currents to B
        dv = (dt*n>tlastB + tref)*(0.04*vB[:,n]**2 + 5*vB[:,n] + 140 - uB[:,n] + IB)/C
        du = (a*(b*(vB[:,n]-vr)-uB[:,n]))
        vB[:,n+1] = vB[:,n] + dv*dt
        uB[:,n+1] = uB[:,n] + du*dt
        idxB = np.argwhere(vB[:,n+1]>=vpeak)
        if len(idxB)>0:
            spike_times = np.ones(len(idxB))*n*dt
            stB.extend(spike_times)
            spike_indices = idxB[:,0]
            siB.extend(spike_indices)
        tlastB = tlastB + (dt*n-tlastB)*(vB[:,n+1]>=vpeak)
        IPSCB = IPSCB*np.exp(-dt/tr) + hB*dt
        hB = hB*np.exp(-dt/td) + JDE1*(len(idxE1)>0)/(tr*td)
        uB[:,n+1] = uB[:,n+1] + d*(vB[:,n+1]>=vpeak)
        vB[:,n+1] = vB[:,n+1] + (vreset-vB[:,n+1])*(vB[:,n+1]>=vpeak)

    stpos = np.array(stpos)
    sipos = np.array(sipos)
    stneg = np.array(stneg)
    sineg = np.array(sineg)
    stE1= np.array(stE1)
    siE1 = np.array(siE1)
    stB = np.array(stB)
    siB = np.array(siB)
    stin = np.hstack((stpos,stneg))
    siin = np.hstack((sipos,sineg+Nin))
    
    st = [stin,stE1,stB]
    si = [siin,siE1,siB]
    print('\tsim done')
    return [tvec, sos, st, si]


    
    