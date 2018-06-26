import numpy as np
import pylab as plt

def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax



def movie(TITLE, MP4_NAME, total=600, sub=100, fps=24):
    """
    """
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation

    #switch on if you want dark background
    #plt.style.use('dark_background')
    
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=TITLE, artist='Matplotlib',
                    comment=TITLE)
    writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=6000)
    Name_mp4=MP4_NAME
                                                        
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()
    P1=fig.add_subplot(1,1,1)
    plt.subplots_adjust(left=0.12, right=0.98, bottom=0.09,top=0.99)

    with writer.saving(fig, Name_mp4, 250):
        for i in range(100):
            print i 
            if i!=0:
                ax.cla()
                plt.subplot(1,1,1)

            X = np.random.normal(size=1000)
            Y = np.random.normal(size=1000)
            plt.scatter(X,Y,c='b',s=20,lw=0)
            plt.xlabel('X',fontsize=20)
            plt.ylabel('Y',fontsize=20)
            plt.xlim(-10,10)
            plt.ylim(-10,10)

            subpos = [0.73,0.08,0.25,0.25]
            subax1 = add_subplot_axes(P1,subpos)
            # if you want to get the dark background 
            #subax1.patch.set_facecolor('black')
            X = np.random.normal(scale = 3, size=1000)
            Y = np.random.normal(size=1000)
            subax1.scatter(X,Y,c='r',lw=0,s=10)
            subax1.set_xlim(-10,10)
            subax1.set_ylim(-10,10)
            subax1.set_xlabel('X',fontsize=20)
            subax1.set_ylabel('Y',fontsize=20)
            
            
            writer.grab_frame()

if __name__ == '__main__':

    movie('Francis','Test_movie.mp4')
