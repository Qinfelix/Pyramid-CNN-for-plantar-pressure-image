import csv  
import numpy as np
import matplotlib.pyplot as plt
import glob2 as glob
import pandas as pd
import torch
from torch import nn
class Footscan_reader():

    def __init__(self):
        self=self
        pass

    # crop the video to the min size
    def more_effective(self,input):
        (timestep,height,width)=np.shape(input)
        valid_h=[0,0]
        valid_w=[0,0]
        for i in range(height):
            if np.mean(input[:,i,:])>0:
                valid_h[0]=i
                break
        for i in range(height):
            if np.mean(input[:,height-1-i,:])>0:
                valid_h[1]=height-1-i
                break
        for i in range(width):
            if np.mean(input[:,:,i])>0:
                valid_w[0]=i
                break
        for i in range(width):
            if np.mean(input[:,:,width-i-1])>0:
                valid_w[1]=width-i-1
                break
        valid_video=input[:,valid_h[0]:valid_h[1],valid_w[0]:valid_w[1]]
        return valid_video

    def show_video(self,video,filepath,speed):
        if(video.ndim==3):
            print('开始播放')
            for i in range(video.shape[0]):
                if i%speed==0:
                    plt.cla()
                    plt.imshow(video[i,:,:])
                    plt.savefig(filepath)
            print('完成播放')
        else:
            print('视频矩阵维度错误，无法展示')
        plt.ioff()

    def record_frame(self,video,planar,planar_height,planar_width):
        planar=planar.reshape((1,planar_height,planar_width))
        return np.append(video,planar,axis=0)#拼接到全局变量中去

    '''
    the function to read the entireplate planar pressure file
    input: filepath
    output: entireplate(np.array shape=time,height,width) ; count
    '''
    def read_footscan_xls(self,filepath):
        smallflag=0 #the row number record of each frame
        planar=np.array([])#buffer of a pressure image 
        planar_video = np.array([])#buffer of the all video
        planar_width =0 
        header_flag = 1
        count=0
        frame_num=-1
        planar_half_height=20

        with open(filepath,'r',encoding='gbk') as myFile:  
            lines=csv.reader(myFile)  
            for line in lines: 
                #data row
                if len(line)!=0 and header_flag==0:
                    # pure data row
                    if len(line[0])>30: 
                        smallflag+=1
                        line_list = line[0].split('\t')
                        planar_width=len(line_list)
                        
                        if line_list[0]=='NaN': 
                            smallflag=0

                        else:
                            for i in range(len(line_list)):
                                line_list[i]=float(line_list[i])
                            line_np=np.array(line_list)
                            planar=np.append(planar,line_np)
                        
                        
                    #'Frame --' row meaning the beginning of a new image
                    else:
                        smallflag=0
                        planar=np.array([])
                        frame_num+=1


                #blank row
                elif len(line)==0 :
                    header_flag=0 
                    # buffer is not empty, so we should cut the image around the cop and joint it to the video
                    if(smallflag>0):
                        planar=planar.reshape(1,smallflag,planar_width)
                        # joint to the video
                        if len(planar_video)==0:
                            #beginning frame
                            planar_video=np.zeros((1,smallflag,planar_width)) 
                            planar_video = np.append(planar_video,planar,axis=0)
                        else:
                            planar_video = np.append(planar_video,planar,axis=0)
                        #judge whether is in the gait:
                        # if sum(planar)


                # get the count number by analyzing the info rows
                if header_flag ==1 : 
                    #first row is enough
                    if line[0][0]=='E':
                        for i in range(len(line[0])):
                            if line[0][i].isdigit()==True:
                                try:
                                    tenornot=line[0][i+1].isdigit()
                                    if tenornot==True:
                                        count=int(line[0][i:i+2])
                                    else:
                                        count=int(line[0][i])
                                except:
                                    count=int(line[0][i])
                                break
                        #get the cop sequence of this count from the dictionary of the whole visit
                        # cop_seq=cop_dic[count]
                            
        return planar_video,count
        

    '''
    the function to read the cop file
    input: cop filepath
    output: cop_Seq(nd_array,shape-time*2) count 
    hypthesis: NaN is considered as 0
    '''
    def read_cop(self,cop_path):
        count=0
        cop_seq=[]
        with open(cop_path,'r',encoding='gbk') as myFile: 
            lines=csv.reader(myFile)
            headerflag=1
            for line in lines:
                #blank row meaning the end of the info header and begin data reading
                if len(line)==0:
                    headerflag=0
                else:
                    #meaning this is the first row, we should extract the count number from it
                    if headerflag==1:
                        if line[0][0]=='E':
                            for i in range(len(line[0])):
                                if line[0][i].isdigit()==True:
                                    try:
                                        tenornot=line[0][i+1].isdigit()
                                        if tenornot==True:
                                            count=int(line[0][i:i+2])
                                        else:
                                            count=int(line[0][i])
                                    except:
                                        count=int(line[0][i])
                                    break
                    #the data row
                    else:
                        line_list=line[0].split('\t')
                        if len(line_list)!=3:
                            print('there is wrong row with incorrect length')
                        else:
                            if line_list[1]=='NaN':
                                cop_seq.append((0,0))
                            else:
                                cop_seq.append((float(line_list[1]),float(line_list[2])))
        
        return np.array(cop_seq),count


    #data argumentation to a entire plate data
    def video_augment_padding(self,input,target_height,target_width):
        if input.ndim!=3:
            print('zeropading输入的矩阵维度错误')
        else:
            effec=self.more_effective(input)
            (timestep,height,width)=np.shape(effec)
            video_list=[]
            #different i means different horitional position --seperately:middle,left,right
            pad_h=target_height-height
            pad_w=target_width-width
            

            for i in [2,8,1.2]:
                pad_top=int(pad_h/2)
                pad_bottom=pad_h-pad_top

                pad_left=int(pad_w/i)
                pad_right=pad_w-pad_left
                
                padder =nn.ZeroPad2d((pad_left,pad_right,pad_top,pad_bottom))
                output = torch.zeros((timestep,target_height,target_width))
                for i in range(timestep-1):
                    output[i,:,:]= padder(torch.tensor(effec[i,:,:]))
                video_list.append(output)
            
            #different j means different vertical position --seperately:top,bottom
            for j in [8,1.2]:
                pad_top=int(pad_h/j)
                pad_bottom=pad_h-pad_top

                pad_left=int(pad_w/2)
                pad_right=pad_w-pad_left
                
                padder =nn.ZeroPad2d((pad_left,pad_right,pad_top,pad_bottom))
                output = torch.zeros((timestep,target_height,target_width))
                for i in range(timestep-1):
                    output[i,:,:]= padder(torch.tensor(effec[i,:,:]))
                video_list.append(output)

        return video_list
    
    #对视频时间轴上做normalization，在步态周期平均抽norm_length个点数，线性插值得到norm之后的数据
    def normalize_video(self,video,norm_length):
        period=video.shape[0]-1
        if video.ndim!=3 :
            raise TypeError('cut video is not in correct shpape')
        #我都默认原始的采样频率是1，依次建立时间轴
        delta=period/(norm_length-1)
        height=video.shape[1]
        width=video.shape[2]
        ans=torch.zeros((norm_length,height,width))
        for i in range(norm_length):
            t=i*delta
            left_point=t//1
            left_distance=t-left_point
            right_distance=1-left_distance
            if int(left_point)==period:
                ans[i,:,:]=video[int(left_point),:,:]
            else:
                ans[i,:,:]=video[int(left_point),:,:]*right_distance + video[int(left_point)+1,:,:]*left_distance
        return ans,delta
    
    '''
    the function which can extract the gait cycle and cut the planar to more focus area
    , and then normalize it to 100 frame of a gait cycle
    input: original video(ndarray format)
    output: normalized video(tensor format)
    '''
    def extract_gait_cycle(self,video):
        alpha=0.2
        nor_height=180
        nor_length=100
        l_force_sum=[]
        r_force_sum=[]
        l_flag=0
        r_flag=0
        l_heel_in=0
        r_heel_in=0
        l_heel_out=0
        r_heel_out=0
        
        for i in range(video.shape[0]):
            l_force_sum.append(sum(video[i,:,0:32].reshape(-1)))
            r_force_sum.append(sum(video[i,:,32:-1].reshape(-1)))
        
        #recognize the heel in and out time 
        for j in range(len(l_force_sum)):
            if l_flag==0 and l_force_sum[j]>alpha*np.mean(l_force_sum):
                l_heel_in=j
                l_flag=1
            if l_flag==1 and l_force_sum[j]<alpha*np.mean(l_force_sum):
                l_heel_out=j
                break
        for s in range(len(r_force_sum)):
            if r_flag==0 and r_force_sum[s]>alpha*np.mean(r_force_sum):
                r_heel_in=s
                r_flag=1
            if r_flag==1 and r_force_sum[s]<alpha*np.mean(r_force_sum):
                r_heel_out=s
                break
        
        #recognize the first leg L/R
        if l_heel_in<r_heel_in:
            gait_begin=l_heel_in
            gait_end=r_heel_out
            print('left first gait')
            first_leg_side='L'
        else:
            gait_begin=r_heel_in
            gait_end=l_heel_out
            print('right first gait')
            first_leg_side='R'

        if gait_end==0 or gait_begin==0:
            print('invalid planar pressure due to gait offset')
            return 0,0,0
        
        #cut the entire plate to the middle part
        for h in range(video.shape[1]):
            if sum(video[gait_begin,h,:])>5:
                upper_height=max(h-15,0)
                break
        if upper_height+nor_height>=video.shape[1]:
            print('invalid planar pressure with too big upper height')
            return 0,0,0
        else:
            norm_video=np.zeros((gait_end-gait_begin,nor_height,video.shape[2]))
            for t in range(gait_end-gait_begin):
                norm_video[t,:,:]=video[t+gait_begin,upper_height:(upper_height+nor_height),:]

            norm_video,delta=self.normalize_video(torch.from_numpy(norm_video),nor_length)
            return norm_video,delta,first_leg_side

    '''
    获得某一个具体的data的接口
    para—— name:'患者的名字',week:'12/24/36/48',injuryside='L'or'R','augment'='1'or'0'
    return——train_x,week,count,delta_list 为tensor,维度分别为(batch,timestep,channel=1,height,width)和(batch)
    路径定义在函数内部，不作为参数传入
    '''
    def get_data(self,name,week,augment,injuryside):
        input_list = []
        week_list=[]
        count_list=[]
        visit_number=name+str(week)+'W'
        footscan_path=[]
        #患侧为左侧代号为-1，右侧代号为1
        if injuryside=='R':
            injury_number=1
        else:
            injury_number=-1
        augment_size=9

        #开始读取
        print('searching the planar pressure file of No.{} visit'.format(visit_number))
        footscan_path_withcop = glob.glob(r'M:\\WTM\\qinyue\\data\\dynamic\\entireplate\\{}\\*.xls'.format(visit_number))
        #从所有文件中分辨cop和足压数据
        for path in footscan_path_withcop:
            if path[-5]=='f':
                footscan_path.append(path)
        file_number = len(footscan_path)

        if file_number == 0:
            print('No measurement in this visit')
            print('\n')
            return 0,0,0,0
            
        else:
            #读取的统计信息输出
            print('there are {} measurements of this visit'.format(file_number))

            print('reading these measurement files')
            #左脚伤为week为-12/-24/-36/-48 ，右脚伤为正数，总共有八类
            for path in footscan_path:
                entireplate,count=self.read_footscan_xls(path)
                pressure_aver=np.mean(entireplate)
                if pressure_aver!=0:
                    input_list.append(entireplate)
                    count_list.append(count)
                else:
                    print('{} pressure file have all-zero data which is invalid'.format(path))

            #转化为tensor+数据增强
            norm_length=100
            input_tensor_list=[]
            delta_list=[]
            valid_count=[]
            first_leg_side_list=[]
            
            if len(count_list)==0:
                print('no valid data in {} measurements'.format(file_number))
                return 0,0,0,0
            else:
                if augment == 0:
                    print('No data argument')
                    for i in range(len(input_list)):
                        video_norm,delta,first_leg_side=self.extract_gait_cycle(input_list[i])
                        if delta!=0:
                            input_tensor_list.append(video_norm)
                            delta_list.append(delta)
                            week_list.append(int(week)*injury_number)
                            valid_count.append(count_list[i])
                            first_leg_side_list.append(first_leg_side)

                else:
                    print('data argument is processing')
                    # for entireplate in input_list:
                    #     entireplate_tensor=torch.from_numpy(entireplate)
                        # augment_list=augment_paading(entireplate)
                        # for i in range(len(augment_list)):
                        #     video_norm,delta=normalize_video(entireplate,norm_length)
                        #     input_tensor_list.append(video_norm)
                        #     delta_list.append(delta)
                    
                train_x=torch.stack(input_tensor_list) 
                train_x=torch.unsqueeze(train_x,2)
                print('the planar pressure reading is FINISHED, the train input shape is ',train_x.shape)

                
                if augment==1:
                    week=np.array(week_list)
                    week=np.expand_dims(week,1).repeat(augment_size,axis=1)
                    week=week.reshape(-1,1)
                    week=torch.tensor(week,dtype=torch.long)   

                    count=np.array(valid_count)
                    count=np.expand_dims(count,1).repeat(augment_size,axis=1)
                    count=count.reshape(-1,1)
                    count=torch.tensor(count,dtype=torch.long)  
                else:
                    week=torch.tensor(week_list,dtype=torch.long)  
                    count=torch.tensor(valid_count,dtype=torch.long)  


                return train_x,week,count,delta_list,first_leg_side_list



class OUTPUT_INFO():
    def __init__(self,gaitdata,group,delta,norm_data):
        self.raw_data=gaitdata.data
        self.group=group
        self.delta=delta
        self.norm_data=norm_data
        self.name=gaitdata.name
        self.count=gaitdata.count
        self.category=gaitdata.category


class GAIT_DATA():
    def __init__(self,category,name,week,count,data):
        self=self
        self.category=category
        self.name=name
        self.week=week
        self.count=count
        self.data=data

class gait_data_reader():
    def __init__(self):
        self=self
        pass


    '''
    统计某一读取批次的步态数据，计算此批数据共包含了多少次回访，并返回这些回访信息，便于匹配相应的足压数据
    '''
    def get_visit_list(self,gait_data_list,info_dic):
        visit_list=[]
        name_list=[]
        category_list=[]
        for gaitdata in gait_data_list:
            visit=(info_dic.get(gaitdata.name),str(gaitdata.group))
            category=gaitdata.category
            #判断名字是否出现过
            if visit[0] in name_list:
                pass
            else:
                name_list.append(visit[0])
            #判断此次回访是否出现过
            if visit in visit_list:
                pass
            else:
                visit_list.append(visit)
            if len(category)!=6:
                print(visit,gaitdata.count,category)


        print('该部分数据中共有 {} 位患者，平均每位患者有{}次回访，共计{}次,平均每次回访测试了{}趟，共计{}趟'.format(len(name_list),len(visit_list)/len(name_list),len(visit_list),len(gait_data_list)/len(visit_list),len(gait_data_list)))
        print('\n \n')
        return visit_list


    '''
    读取单个topic的gaitdata文件的函数,输入是文件路径,输出是一个list,list的每一个量都是特定一次回访的特定一趟的该主题下的所有种类的数据（全部集成在一个gaitdata结构体中，category是一个list，data的列数目就是模拟量种类数目*3）
    '''
    def read_file(self,filepath):
        file=open(filepath,encoding='gbk')
        #先读出所有数据
        all_rows=[]
        for line in file.readlines():
            row=line.split('\t')
            all_rows.append(row)
        row_length=len(all_rows[0])
        row_num=len(all_rows)

        #对前5行进行操作，提取出D1W04+T1+W1+data_name(ex.pelvsi cog)这四个信息，每一项都是一个tuple（name，week，count,category）
        info_row=all_rows[0]
        info_list=[]
        category_list=[]
        file_begin_index=[1]
        last_file_info=info_row[1]
        for i in range(1,row_length,3):

            this_info=info_row[i]
            #category一定在第二行
            category=all_rows[1][i]

            
            #记录每一个文件的数据的开始(上一个步态文件的全部数据均读取完毕)
            if this_info!=last_file_info:
                file_begin_index.append(i)
                #name的提取没有什么大问题_基本都在第一行的前五个
                name=last_file_info[0:5]
                #week不一定都在一个固定的位置，但是一定是name后面的第一个数字
                for j in range(5,15):
                    if last_file_info[j].isdigit()==True:
                        week=int(last_file_info[j:j+1])
                        break
                #count一定在最后的.c3d前面一个，注意区分两位数和一位数就好
                if last_file_info[-6]=='1':
                    count=int(last_file_info[-6:-4])
                else:
                    count=int(last_file_info[-5])
                info_list.append((name,week,count,category_list))
                category_list=[]

            #最后一组
            if i==row_length-3:
                category_list.append(category)
                #name的提取没有什么大问题_基本都在第一行的前五个
                name=last_file_info[0:5]
                #week不一定都在一个固定的位置，但是一定是name后面的第一个数字
                for j in range(5,15):
                    if last_file_info[j].isdigit()==True:
                        week=int(last_file_info[j:j+1])
                        break
                #count一定在最后的.c3d前面一个，注意区分两位数和一位数就好
                if last_file_info[-6]=='1':
                    count=int(last_file_info[-6:-4])
                else:
                    count=int(last_file_info[-5])
                info_list.append((name,week,count,category_list))
                category_list=[]


            last_file_info=this_info

            category_list.append(category)
               
        print('在文件路径{}下，共读取了{}个步态文件'.format(filepath,len(info_list)))


        #对后面的数据行进行处理，集成到一个ndarrary当中，大小帧数*3（3代表xyz三个方向）
        data_list=[]

        for i in range(len(file_begin_index)):
            #读取一个文件的所有data
            begin_index=file_begin_index[i]
            if i==len(file_begin_index)-1:
                end_index=row_length
            else:
                end_index=file_begin_index[i+1]
            
            column_num=end_index-begin_index
            buffer=[]
            for row_index in range(5,row_num):
                #先检测后期出现无数据行
                if row_index>1500 and all_rows[row_index][begin_index:end_index]==['']*column_num:
                    break
                else:
                    buffer.extend(all_rows[row_index][begin_index:end_index])
            #下面将读取的数据都转化为浮点数，如果遇到数据缺失，就视为和上一帧数据一样
            float_buffer=[]
            for i in range(len(buffer)):
                if buffer[i]=='':
                    buffer[i]=0
                try:
                    float(buffer[i])
                    float_buffer.append(float(buffer[i]))
                except:
                    try:
                        a=float_buffer[i-column_num]
                        float_buffer.append(a)
                    except:
                        float_buffer.append(0)
                        print('{}文件中出现错误，此时的gaitdata 表格中出现空格数据，而且前面没有可参考值'.format(filepath))
            data=np.array(float_buffer,ndmin=1)
            data=data.reshape(-1,column_num)
            data_list.append(data)


        
        #把上面的两个集成在一个结构体内
        gait_data_list=[]
        for i in range(len(info_list)):
            if len(info_list[i][3])==6:
                gaitdata=GAIT_DATA(info_list[i][3],info_list[i][0],info_list[i][1],info_list[i][2],data_list[i])
                gait_data_list.append(gaitdata)
            else:
                print('在{}中出现残缺文件，被删除'.format(filepath))

        return gait_data_list
    
    '''
    该函数就是对目前已有的所有数据进行读取,返回一个list,list的元素是GAITDATA结构体,每一个元素是特定一次回访（name，week、count唯一确定）与该主题（all_angle,grf_exp）相关的所有模拟量的集合——此时的category是一个list，里面集合了包含的所有模拟量的种类；data是一个【m,3*n】m是帧数，n是该主题下的模拟量种类
    此时路径定义在函数内部,并不作为参数传入

    '''
    def get_data(self,data_content):
        print('开始读取{}数据'.format(data_content))
        gaitdata_filepath=glob.glob(r'M:\\WTM\\qinyue\\data\\dynamic\\gait\\*\\*\\{}.txt'.format(data_content))
        gaitdata_list=[]
        for filepath in gaitdata_filepath:
            gaitdata=self.read_file(filepath)
            gaitdata_list.extend(gaitdata)

        
        print('读取完毕，共读取{}个步态文件中的{}数据'.format(len(gaitdata_list),data_content))
        print('\n','*'*50,'\n')
        return gaitdata_list
    

    def read_motsto(self,filepath):
        file=open(filepath,encoding='gbk')
        #先读出所有数据
        all_rows=[]
        for line in file.readlines():
            row=line.split('\t')
            all_rows.append(row)
        row_num=len(all_rows)
        enheader_index=0
        #先获得enheader在哪一行
        for i in range(row_num):
            if all_rows[i][0]=='endheader\n':
                enheader_index=i
                break
        category_list=all_rows[enheader_index+1]
        column_num=len(category_list)



        buffer=[]
        for row_index in range(enheader_index+2,row_num):
            #先检测后期出现无数据行
            if row_index>100 and all_rows[row_index][:]==['']*column_num:
                break
            else:
                buffer.extend(all_rows[row_index][:])

        #下面将读取的数据都转化为浮点数，如果遇到数据缺失，就视为和上一帧数据一样
        float_buffer=[]
        for i in range(len(buffer)):
            if buffer[i]=='':
                buffer[i]=0
            try:
                float(buffer[i])
                float_buffer.append(float(buffer[i]))
            except:
                try:
                    a=float_buffer[i-column_num]
                    float_buffer.append(a)
                except:
                    float_buffer.append(0)
                    print('{}文件中出现错误，此时的gaitdata 表格中出现空格数据，而且前面没有可参考值'.format(filepath))
        data=np.array(float_buffer,ndmin=1)
        data=data.reshape(-1,column_num)
        
        gaitdata=GAIT_DATA(category_list,'unknown','unknown','unkown',data)
        return gaitdata
    
    def get_heel_inout(self,ankle_moment):
        alpha=0.35
        l_heel_in=0
        l_heel_out=1
        l_flag=0
        for i in range(len(ankle_moment)):
            if l_flag==0 and abs(ankle_moment[i])>-alpha*np.mean(ankle_moment):
                l_heel_in=i
                l_flag=1
            if ankle_moment[i]<0.5*np.min(ankle_moment):
                l_flag=2
            if l_flag==2 and ankle_moment[i]>alpha*np.mean(ankle_moment):
                l_heel_out=i
                break
        return l_heel_in,l_heel_out

    def normlize_moment(self,gaitdata,norm_length):
        data=gaitdata.data
        lankle=data[:,0].reshape(-1)
        rankle=data[:,9].reshape(-1)
        l_heel_in,l_heel_out=self.get_heel_inout(lankle)
        r_heel_in,r_heel_out=self.get_heel_inout(rankle)
        if min(l_heel_in,l_heel_out,r_heel_in,r_heel_out)==0:
            visit=gaitdata.name+'W'+str(gaitdata.week)+'T'+str(gaitdata.count)
            raise ValueError('ZERO heel in or out time of the test No.{}'.format(visit))
        if l_heel_in<r_heel_in:
            first_legside='L'
            gait_begin=l_heel_in
            gait_end=r_heel_out

        else:
            first_legside='R'
            gait_begin=r_heel_in
            gait_end=l_heel_out
        ans=torch.zeros((norm_length,data.shape[1]))
        delta=(gait_end-gait_begin)
        delta=delta/norm_length
        for i in range(norm_length):
            t=gait_begin+i*delta
            left_point=t//1
            left_distance=t-left_point
            right_distance=1-left_distance
            ans[i,:]=torch.from_numpy(data[int(left_point),:]*right_distance + data[int(left_point)+1,:]*left_distance)
        return ans,delta