import re
def get_regular_result(lable_2_title, pred_label, s):
    pred_title = lable_2_title[pred_label]
    title2label = {title: label for label, title in lable_2_title.items()}
    if pred_title == '正常文本':
        for key_word in ['妈的','老赖','屁用','死猪不怕开水烫','垃圾','我操','做梦','狗逼']:
            if key_word in s:
                pred_title = '违规惩处-不当言辞'
                break 
        if re.search(r'睁眼.*瞎话', s):
            pred_title = '违规惩处-不当言辞'
    
    if pred_title == '业务技能规范-拖车':
        if re.search(r'(没有|没|不)[^，。]*[拖收拉拽].*车',s):
            pred_title = '正常文本'
    elif (re.search(r'收[^，。]*车(?!(的话|吗))',s) or \
        re.search(r'车[^，。]*(拖了|拖走了)(?!(的话|吗))',s) or \
        re.search(r'车[^，。他她]*收了(?!(的话|吗))',s) or \
        re.search(r'拖[^，。]?车了',s) or \
        re.search(r'车[^，。他她]*收走(?!(的话|吗))',s) or \
        re.search(r'车[^不]*会.*收走',s)) and \
        re.search(r'(没有|没|不)[^，。]*[拖收拉拽].*车',s) == None:
        pred_title = '业务技能规范-拖车'

    if pred_title == '正常文本' or pred_title == '违规惩处-私自承诺减免':
        if (re.search(r'^[^申请]*(减掉|降低|减少|免|少交|减免).*(违约金|本金|利息|钱|一些|一部分|一点)[^，]*[^吗]',s) or \
            re.search(r'^[^申请]*(违约金|本金|利息|钱).*(减掉|降低|减少|免|少交|减免)',s) or \
            re.search(r'^[^申请]*(减掉|降低|减少|免|少交|减免).*(违约金|本金|利息|钱)',s)) and \
            re.search(r'(怎么|如何|怎样|咋|没有办法|不|应该可以|试着)[^，。]*(减掉|降低|减少|免|少交|减免)',s) == None and \
            re.search(r'(减掉|降低|减少|免|少交|减免).*(违约金|本金|利息|钱)[^，。]*吗',s) == None and \
            re.search(r'(减掉|降低|减少|免|少交|减免)*(不了|不来|不掉)',s) == None and \
            '无法' not in s and '没办法' not in s and '没法' not in s and ('不能' not in s or '能不能' in s) and '申请' not in s \
            and '报备' not in s and '谈一下' not in s:
            pred_title = '违规惩处-私自承诺减免'
        else:
            pred_title = '正常文本'
    if pred_title == '正常文本':
        if (
            re.search(r'(去带|去贷|去借|看看|找找|想想).*(别的|其他|另外).*贷款',s) or \
            re.search(r'(去带|去贷|去借).*(信用卡|支付宝|借呗|花呗|京东|白条)',s) or \
            re.search(r'(信用卡|支付宝|借呗|花呗|京东|白条).*(找钱|来补|贷款|先还|借钱|取钱|周转|调整|转换|套现|套钱)',s) or \
            re.search(r'以贷养贷',s) or re.search(r'美团.*贷款',s) or \
            re.search(r'(信用卡|支付宝|借呗|花呗|京东|白条|银行卡).*(带钱|贷钱|去带|去贷|去借|借啊|借吗)',s)
            ) and re.search(r'(怎能|怎么能|不能|别去|不要|如果|禁止)',s) == None:
            pred_title = '业务技能规范-逼迫其通过非法途径筹集资金'

    if re.search(r'(爆|公布|发|挂|公开).*(照片|个人信息)',s) or \
        re.search(r'(照片|个人信息).*(爆|公布|发|挂|公开)',s):
        pred_title = '违规惩处-将借款人信息公布在公开信息平台'

    if re.search(r'(家里|家人|爸妈|亲戚|朋友|同事|儿子|女儿|儿女|老婆|什么人)[^，。]*(打一遍|会知道)',s):
        pred_title = '违规惩处-爆通讯录'

    if pred_title in ['客户未知意图', '服务态度差', '打断客户说话', '向客户安抚致歉', '耐心倾听客户说话']:
        if re.search(r'(对不起|歉意|抱歉|别担心|我们的问题|我们的原因)', s):
            pred_title = '向客户安抚致歉'
        elif re.search(r'(听我说|别讲话|别说)',s):
            pred_title = '服务态度差'
        elif re.search(r'(耐心听|我在听|慢慢说)',s):
            pred_title = '耐心倾听客户说话'
        if re.search(r'(打断一下|插个嘴|先听我说)',s):
            pred_title = '打断客户说话'

    if pred_title == '业务技能规范-承认我方为高利贷、黑社会等':
        if re.search(r'(才是|什么|不是|哪里|谁说|怎么)[^，。]*高利贷', s) or \
        re.search(r'高利贷[^，。]*(什么|的话|吗|？)', s):
           pred_title = '正常文本'
           
    return title2label[pred_title]