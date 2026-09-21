// 前端实跑：用系统 Chrome 打开界面，收集控制台错误，做一遍交互断言，并截图。
// 用法（在项目根目录执行）：
//   NODE_PATH="C:/Users/gaore/.workbuddy-ai/binaries/node/workspace/node_modules" \
//     node verify_ui.js [截图输出路径]
const path = require('path');
const NODE_WS = 'C:/Users/gaore/.workbuddy-ai/binaries/node/workspace/node_modules';
const { chromium } = require(path.join(NODE_WS, 'playwright-core'));

const URL = 'http://127.0.0.1:7861/';
const SHOT = process.argv[2] || path.join(__dirname, '_ui.png');

let pass = 0, fail = 0;
function check(label, ok, extra) {
  if (ok) { pass++; console.log('  OK  ' + label + (extra ? '  ' + extra : '')); }
  else { fail++; console.log('  BAD ' + label + (extra ? '  ' + extra : '')); }
}

(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage'],
  });
  const page = await browser.newPage({ viewport: { width: 760, height: 1400 } });

  const errors = [], warnings = [];
  page.on('console', m => {
    const t = m.type();
    if (t === 'error') errors.push(m.text());
    else if (t === 'warning') warnings.push(m.text());
  });
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));

  try {
    await page.goto(URL, { waitUntil: 'load', timeout: 30000 });
    // boot() 是异步的，等预设渲染出来
    await page.waitForFunction(
      () => document.querySelectorAll('#presets .chip').length > 0, { timeout: 20000 });
    await page.waitForTimeout(600);

    console.log('\n[1] 结构');
    const cards = await page.$$eval('.num', ns => ns.map(n => n.textContent));
    check('5 张编号卡片', JSON.stringify(cards) === JSON.stringify(['1','2','3','4','5']),
      JSON.stringify(cards));
    check('预设已渲染', await page.$$eval('#presets .chip', n => n.length) >= 13,
      (await page.$$eval('#presets .chip', n => n.length)) + ' 个');
    check('风格组合器·语言', await page.$$eval('#b-lang .bchip', n => n.length) === 6);
    check('风格组合器·流派下拉', await page.$$eval('#b-genre option', n => n.length) >= 50,
      (await page.$$eval('#b-genre option', n => n.length)) + ' 项');
    check('风格组合器·乐器', await page.$$eval('#b-inst .bchip', n => n.length) === 26,
      (await page.$$eval('#b-inst .bchip', n => n.length)) + ' 件');
    check('合奏编制下拉已渲染', await page.$$eval('#b-ensemble option', n => n.length) === 5);
    check('调性下拉已渲染', await page.$$eval('#b-key option', n => n.length) >= 12);
    check('拍号下拉已渲染', await page.$$eval('#b-meter option', n => n.length) >= 7);
    check('采样滑块 top_k 已渲染', await page.$('#a-topk') !== null);
    check('采样滑块 penalty_window 已渲染', await page.$('#a-pwin') !== null);
    check('创作风格 4 档', await page.$$eval('#creative .chip', n => n.length) === 4);
    check('规划方式 4 项（含跟随默认）',
      await page.$$eval('#plan .chip', n => n.length) === 4);
    check('时长 6 档', await page.$$eval('#durations .dur', n => n.length) === 6);
    check('一次出几首 3 档', await page.$$eval('#batch .dur', n => n.length) === 3);
    check('默认出 1 首', await page.$$eval('#batch .dur.on', n => n.length) === 1);
    check('批量结果容器存在', await page.$('#batchlist') !== null);
    check('单首结果容器存在', await page.$('#singlewrap') !== null);
    check('高级设置默认收起',
      await page.$eval('#adv', e => e.style.display) === 'none');
    // 页面会恢复上一次的生成结果；有谱时乐谱面板本来就该露出来
    const job = await page.evaluate(async () => await (await fetch('/api/job')).json());
    const sb = await page.$eval('#scorebox', e => e.style.display);
    if (job.status === 'done' && job.has_score) {
      check('乐谱面板随上次结果恢复显示', sb === 'block', 'job.has_score=true');
    } else {
      check('乐谱面板默认隐藏', sb === 'none', 'job.status=' + job.status);
    }
    check('状态条有显卡信息',
      (await page.$eval('#status', e => e.textContent)).includes('RTX'));

    console.log('\n[2] 交互');
    // 点一个预设
    await page.click('#presets .chip:nth-child(3)');
    const st1 = await page.$eval('#style', e => e.value);
    check('点预设填风格', st1.length > 10, st1.slice(0, 40) + '…');

    // 点组合器积木：语言→粤语，情绪→忧郁，乐器→钢琴
    await page.click('#b-lang .bchip:nth-child(2)');
    await page.click('#b-mood .bchip:nth-child(2)');
    await page.click('#b-inst .bchip:nth-child(1)');
    await page.selectOption('#b-genre', 'City Pop');
    const st2 = await page.$eval('#style', e => e.value);
    check('组合器覆盖预设描述', st2 !== st1, st2);
    check('描述含粤语', st2.includes('Cantonese'));
    check('描述含流派', st2.includes('City Pop'));
    check('描述含情绪', st2.includes('melancholic'));
    check('描述含乐器', st2.includes('piano'));
    check('描述含 BPM', /\d+ BPM/.test(st2));

    // 乐器上限 4 个（MAX_INST）
    await page.click('#b-reset');
    for (let i = 1; i <= 6; i++) {
      const on = await page.$$eval('#b-inst .bchip.on', n => n.length);
      if (on >= 4) break;
      await page.click(`#b-inst .bchip:nth-child(${i})`);
    }
    const onCount = await page.$$eval('#b-inst .bchip.on', n => n.length);
    check('乐器最多选 4 件', onCount === 4, onCount + ' 件');

    // 一次出几首：点 4 首，高亮和提示文案都要跟着走
    await page.click('#batch .dur:nth-child(3)');
    check('点 4 首后高亮切换',
      (await page.$eval('#batch .dur:nth-child(3)', e => e.classList.contains('on')))
      && (await page.$$eval('#batch .dur.on', n => n.length)) === 1);
    check('批量提示文案更新',
      (await page.$eval('#batchhint', e => e.textContent)).includes('4 首'));
    await page.click('#batch .dur:nth-child(1)');
    check('切回 1 首',
      (await page.$$eval('#batch .dur.on', n => n.length)) === 1
      && (await page.$eval('#batch .dur:nth-child(1)', e => e.classList.contains('on'))));

    // 批量结果区渲染：直接喂假数据，验证 4 首能列出来、单首要能切回来
    await page.evaluate(() => {
      showResult({
        status: 'done', seconds: 60, elapsed: 300, seed: 100,
        dir: 'x', has_score: false,
        batch_done: [
          { audio_url: '/api/audio?dir=a', audio_name: 'a.flac', seconds: 60, elapsed: 300, seed: 100 },
          { audio_url: '/api/audio?dir=b', audio_name: 'b.flac', seconds: 59, elapsed: 290, seed: 101 },
          { audio_url: '/api/audio?dir=c', audio_name: 'c.flac', seconds: 61, elapsed: 310, seed: 102 },
          { audio_url: '/api/audio?dir=d', audio_name: 'd.flac', seconds: 60, elapsed: 305, seed: 103 },
        ],
      });
    });
    check('批量结果渲染 4 个播放器',
      (await page.$$eval('#batchlist audio', n => n.length)) === 4,
      (await page.$$eval('#batchlist audio', n => n.length)) + ' 个');
    check('批量时隐藏单首播放器',
      (await page.$eval('#singlewrap', e => e.style.display)) === 'none');
    check('标题显示首数',
      (await page.$eval('#resultlabel', e => e.textContent)).includes('4 首'));
    await page.evaluate(() => {
      showResult({ status: 'done', seconds: 60, elapsed: 300, seed: 100,
                   audio_url: '/api/audio?dir=a', audio_name: 'a.flac',
                   dir: 'a', has_score: false, batch_done: [] });
    });
    check('单首结果能切回来',
      (await page.$eval('#singlewrap', e => e.style.display)) === 'block'
      && (await page.$$eval('#batchlist audio', n => n.length)) === 0);

    // 新增：调性 / 拍号写进描述
    await page.selectOption('#b-key', 'A minor');
    await page.selectOption('#b-meter', '3/4');
    const stk = await page.$eval('#style', e => e.value);
    check('选了调性写进描述', stk.includes('in the key of A minor'), stk);
    check('选了拍号写进描述', stk.includes('in 3/4 time'), stk);

    // 新增：采样滑块标签实时更新
    await page.$eval('#a-topk', e => { e.value = 30; e.dispatchEvent(new Event('input')); });
    check('top_k 标签更新', (await page.$eval('#a-topkval', e => e.textContent)) === '30');
    await page.$eval('#a-pwin', e => { e.value = 20; e.dispatchEvent(new Event('input')); });
    check('penalty_window 标签更新',
      (await page.$eval('#a-pwinval', e => e.textContent)) === '20');

    // 新增：合奏编制一键填充乐器 + 写进描述
    await page.click('#b-reset');
    await page.selectOption('#b-ensemble', 'guqin+xiao');
    const ensOn = await page.$$eval('#b-inst .bchip.on', n => n.length);
    check('选合奏自动选中 2 件乐器', ensOn === 2, ensOn + ' 件');
    const stE = await page.$eval('#style', e => e.value);
    check('合奏写进描述', stE.includes('with guqin and xiao'), stE);
    await page.selectOption('#b-ensemble', 'guqin+guzheng+pipa+dizi');
    const ens4 = await page.$$eval('#b-inst .bchip.on', n => n.length);
    check('丝竹乐合奏选中 4 件', ens4 === 4, ens4 + ' 件');
    const stE4 = await page.$eval('#style', e => e.value);
    check('四件套全部拼进描述',
      stE4.includes('with guqin and guzheng and pipa and dizi'), stE4);
    await page.selectOption('#b-ensemble', '');
    check('取消合奏后描述不再含 with 短语',
      !(await page.$eval('#style', e => e.value)).includes('with guqin'));

    // 清空重来
    await page.click('#b-reset');
    const st3 = await page.$eval('#style', e => e.value);
    check('清空重来只剩语言+BPM', !st3.includes('City Pop'), st3);

    // 规划方式
    await page.click('#plan .chip:nth-child(3)');   // 旋律规划
    let hint = await page.$eval('#planhint', e => e.textContent);
    check('点「旋律规划」提示更新', hint.includes('旋律'), hint.slice(0, 30));
    await page.click('#plan .chip:nth-child(4)');   // 直接生成
    hint = await page.$eval('#planhint', e => e.textContent);
    check('点「直接生成」提示更新', hint.includes('最快'), hint.slice(0, 30));

    // 填了外部乐谱 -> 提示自动抬到完整规划
    await page.click('#advtoggle');
    check('展开高级设置',
      await page.$eval('#adv', e => e.style.display) === 'block');
    await page.fill('#a-abc', 'X:1\nK:C\nCDEF');
    hint = await page.$eval('#planhint', e => e.textContent);
    check('有谱时提示会被抬到完整规划', hint.includes('完整规划'), hint.slice(0, 34));

    // 高级设置滑块
    await page.$eval('#a-cfg', e => { e.value = '7.5'; e.dispatchEvent(new Event('input')); });
    check('风格贴合度显示 7.5',
      (await page.$eval('#a-cfgval', e => e.textContent)) === '7.5');
    await page.$eval('#a-rep', e => { e.value = '1.35'; e.dispatchEvent(new Event('input')); });
    check('重复抑制显示 1.35',
      (await page.$eval('#a-repval', e => e.textContent)) === '1.35');
    await page.click('#a-ode .bchip:nth-child(3)');
    check('合成精细度选中 48',
      (await page.$eval('#a-ode .bchip.on', e => e.textContent)).includes('48'));

    // 种子校验：填负数应该被拦
    let alerted = '';
    page.once('dialog', async d => { alerted = d.message(); await d.dismiss(); });
    await page.fill('#a-seed', '-5');
    await page.click('#go');
    await page.waitForTimeout(400);
    check('负数种子被拦下', alerted.includes('非负整数'), alerted);

    console.log('\n[3] 乐谱面板');
    if (job.status === 'done' && job.has_score) {
      await page.click('#scoretoggle');
      await page.waitForFunction(() => {
        const t = document.querySelector('#scoretext').textContent;
        return t && t !== '读取中…';
      }, { timeout: 15000 });
      const abc = await page.$eval('#scoretext', e => e.textContent);
      check('展开后读到乐谱', abc.startsWith('X:'), abc.slice(0, 26).replace(/\n/g, ' '));
      await page.click('#scoreuse');
      const used = await page.$eval('#a-abc', e => e.value);
      check('「拿去当外部乐谱」回填成功', used.startsWith('X:'), used.length + ' 字符');
      const pm = await page.$eval('#plan .chip.on', e => e.textContent);
      check('回填后规划方式切到完整规划', pm.includes('完整规划'), pm);
    } else {
      console.log('  （当前没有带谱的结果，跳过）');
    }

    console.log('\n[4] 控制台');
    // 上面用假数据测批量渲染时，音频 URL 故意指向了不存在的目录，
    // 浏览器会为 <audio src> 报 404。那不是代码问题，过滤掉再检查。
    const realErrors = errors.filter(e => !/status of 404/.test(e));
    check('无 JS 报错', realErrors.length === 0, realErrors.slice(0, 3).join(' | '));
    if (warnings.length) console.log('  （警告 ' + warnings.length + ' 条，忽略）');

    await page.screenshot({ path: SHOT, fullPage: true });
    console.log('\n截图：' + SHOT);
  } catch (e) {
    console.log('\n脚本异常：' + e.message);
    fail++;
    try { await page.screenshot({ path: SHOT, fullPage: true }); } catch (_) {}
  } finally {
    await browser.close();
  }

  console.log('\n通过 ' + pass + ' 项，失败 ' + fail + ' 项');
  process.exit(fail ? 2 : 0);
})();
