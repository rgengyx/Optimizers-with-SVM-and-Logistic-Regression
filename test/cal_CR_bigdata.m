function CR = cal_CR_bigdata(data,label,opts)
%CAL_CR 此处显示有关此函数的摘要
%   此处显示详细说明
len = length(data);
paras = opts.paras;
bar = opts.bar;
label_bar = opts.label_bar;
res_count = 0;

    function out = cal_val(x,paras)
        x_len = length(x);
        out = paras(1:x_len)' * x + paras(x_len+1);
    end
for i = 1:len
    x_now = data(:,i);pred_value = cal_val(x_now,paras);
    if((pred_value - bar) * (label(i) - label_bar) >= 0)%right prediction
        res_count = res_count + 1;
    end
end

%calculate the correctness ratio
CR = res_count / len;
end


