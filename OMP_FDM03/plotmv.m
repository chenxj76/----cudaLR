writerObj=VideoWriter('test.avi');  %// ����һ����Ƶ�ļ������涯��  
writerObj.FrameRate=3;
open(writerObj);                    %// �򿪸���Ƶ�ļ�  
for i = 0:1:50  
    f1 = figure(1);
    hold on
%     colormap(gray(256));
    colormap(jet);
    ap=['ap', num2str(i)];
    T = readtable(ap);
    area=table2array(T);
    imagesc(1:1000,1:1000,area(1:1000,1:1000))%eval: Execute MATLAB expression in text
    axis([1 1000 1 1000])
    colorbar()
    if(i==0)
        title('t=0.02ms');
    else
        title(['t=' num2str(i*10) 'ms']);
    end
    pause(0.1)
    frame = getframe(f1);            %// ��ͼ�������Ƶ�ļ���  
    writeVideo(writerObj,frame); %// ��֡д����Ƶ  
end
close(writerObj); %// �ر���Ƶ�ļ����  