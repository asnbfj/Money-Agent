# Spectrum 2 设计系统文件

页面按 Spectrum 2 设计系统（https://claude.ai/artifact/Q7HSZRbPQoMtg3Kgxp16H5）构建：

- `tokens.css`：由设计系统的 `project/tokens.json` 生成，包含浅色 / 深色两套变量。
- `components.css`：设计系统的组件样式 `project/components/bundle.css`，原样复制。

设计系统更新后，下载新的 `tokens.json` 和 `bundle.css`，然后运行：

```bash
python3 frontend/vendor/spectrum/gen_tokens.py path/to/tokens.json frontend/vendor/spectrum/tokens.css
cp path/to/bundle.css frontend/vendor/spectrum/components.css
```

页面严格按 Spectrum 2 构建，样式（`frontend/index.html` 里的 `<style>`）只引用这些 token。唯一有意的偏离：

- 涨跌按中国市场习惯红涨绿跌，用色相 token `red-900` / `green-900`（文字）和 `red-visual-color` / `green-visual-color`（圆点、图表），不借用 positive / negative 的语义色。

中文字体（Source Han Sans / PingFang / Noto Sans SC）接在 `--font-sans` 的拉丁字体之后，对应设计系统 README 里 `--font-cjk` 的回退顺序。
