// Renders text lines with CoreText (proper complex-script shaping) to a PNG.
// usage: swift render.swift out.png width height  then lines on stdin: "fontName<TAB>size<TAB>text"
import AppKit
import CoreText

let args = CommandLine.arguments
let out = args[1], w = Int(args[2])!, h = Int(args[3])!
let cs = CGColorSpaceCreateDeviceRGB()
let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
var y = 30.0
while let line = readLine() {
  let p = line.components(separatedBy: "\t")
  if p.count < 3 { continue }
  let size = Double(p[1])!
  let font = CTFontCreateWithName(p[0] as CFString, size, nil)
  let attr = NSAttributedString(string: p[2], attributes: [
    NSAttributedString.Key(kCTFontAttributeName as String): font,
    NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(red: 0, green: 0, blue: 0, alpha: 1)])
  let ctl = CTLineCreateWithAttributedString(attr)
  y += size * 1.3
  ctx.textPosition = CGPoint(x: 30, y: Double(h) - y)
  CTLineDraw(ctl, ctx)
  y += size * 0.7
}
let img = ctx.makeImage()!
let rep = NSBitmapImageRep(cgImage: img)
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: out))
