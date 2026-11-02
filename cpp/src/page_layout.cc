#include "page_layout.h"

#include "proto_wire.h"

namespace screenai {
namespace {

constexpr int kPhotoOcrTextDetectionInfoExt = 12365408;  // google_ocr.PhotoOcrTextDetectionInfo.ext
constexpr int kAksaraInternalEntityExt = 108072910;      // aksara.api_internal.InternalEntityExt.ext

pb::Writer CurveFields(const Curve& c) {
  pb::Writer w;  // shared field numbers 1..3 of goodoc.CurvedBoundingBox and ocr.photo.CurvedBoundingBox
  pb::Writer curve;
  for (const auto& [x, y] : c.points) {
    pb::Writer p;
    p.Double(1, x);
    p.Double(2, y);
    curve.Message(1, p);
  }
  w.Message(1, curve);
  w.Double(2, c.thickness);
  if (c.top_to_bottom) w.Bool(3, *c.top_to_bottom);
  return w;
}

// goodoc.BoundingPolygon: rotated_box, or curved_box carrying the rotated box as rotated_bounding_box.
pb::Writer Polygon(const RotatedBox& b);

pb::Writer Box(const RotatedBox& b) {
  pb::Writer w;  // goodoc.RotatedBoundingBox / ocr.photo.BoundingBox share field numbers 1..5
  w.Int(1, b.left);
  w.Int(2, b.top);
  w.Int(3, b.width);
  w.Int(4, b.height);
  w.Float(5, b.angle);
  return w;
}

pb::Writer Entity_(const Entity& e) {
  pb::Writer w;
  if (e.parent >= 0) w.Int(1, e.parent);
  if (e.box) {
    w.Message(2, Polygon(*e.box));
  }
  w.Int(4, static_cast<int>(e.type));
  if (e.content_type) w.Int(5, *e.content_type);
  if (e.orientation || e.writing_direction || e.textline_order) {
    pb::Writer o;
    if (e.orientation) o.Int(1, *e.orientation);
    if (e.writing_direction) o.Int(2, *e.writing_direction);
    if (e.textline_order) o.Int(3, *e.textline_order);
    w.Message(6, o);
  }
  if (e.confidence) w.Float(7, *e.confidence);
  if (!e.languages.empty()) {
    pb::Writer mix;
    for (const auto& l : e.languages) {
      pb::Writer lang;
      lang.Bytes(1, l);
      mix.Message(1, lang);
    }
    w.Message(13, mix);
  }
  if (e.detection_confidence) w.Float(14, *e.detection_confidence);
  if (e.text) w.Bytes(15, *e.text);
  if (e.word_break) {
    pb::Writer b;
    b.Int(1, e.word_break->type);
    if (e.word_break->box) b.Message(2, Box(*e.word_break->box));
    w.Message(16, b);
  }
  if (e.colors) {
    const Colors& c = *e.colors;
    pb::Writer cw;
    cw.Int(1, c.foreground_gray);
    cw.Int(2, c.background_gray);
    if (c.foreground_rgb) cw.Int(3, *c.foreground_rgb);
    if (c.background_rgb) cw.Int(4, *c.background_rgb);
    if (c.foreground_gray_threshold) cw.Int(5, *c.foreground_gray_threshold);
    for (const auto& [x, y] : c.foreground_samples) {
      pb::Writer p;
      p.Int(1, x), p.Int(2, y);
      cw.Message(6, p);
    }
    for (const auto& [x, y] : c.background_samples) {
      pb::Writer p;
      p.Int(1, x), p.Int(2, y);
      cw.Message(7, p);
    }
    w.Message(17, cw);
  }
  if (e.type_confidence) w.Float(22, *e.type_confidence);
  if (e.recognizer_script_id) w.Bytes(23, *e.recognizer_script_id);
  if (e.direction_confidence) w.Float(27, *e.direction_confidence);
  if (e.predicted_direction) w.Int(32, *e.predicted_direction);
  if (e.detection) {
    const DetectionInfo& d = *e.detection;
    pb::Writer dw;
    pb::Writer det_box = Box(d.detection_box);  // ocr.photo.BoundingBox: curve in field 6
    if (d.detection_box.curve) det_box.Message(6, CurveFields(*d.detection_box.curve));
    dw.Message(1, det_box);
    dw.Int(4, d.detection_image_scale);
    dw.Float(9, d.average_raw_detections_height);
    dw.Float(10, d.variance_raw_detections_height);
    dw.Float(11, d.detection_nms_score);
    dw.Int(12, d.num_raw_boxes);
    dw.Float(13, d.sum_raw_boxes_scores);
    dw.Float(14, d.left_padding);
    dw.Float(15, d.right_padding);
    if (d.garbage_prob) dw.Float(16, *d.garbage_prob);
    dw.Bool(17, d.fallback_upscaling);
    w.Message(kPhotoOcrTextDetectionInfoExt, dw);
  }
  if (e.aksara) {
    const AksaraResult& a = *e.aksara;
    pb::Writer result;  // aksara.TextLineResult
    if (a.text) result.Bytes(1, *a.text);
    if (a.right_to_left) result.Bool(19, *a.right_to_left);
    if (a.language) result.Bytes(20, *a.language);
    if (a.confidence) result.Float(21, *a.confidence);
    if (a.predicted_script) result.Bytes(24, *a.predicted_script);
    pb::Writer ext;
    ext.Message(3, result);
    w.Message(kAksaraInternalEntityExt, ext);
  }
  return w;
}

pb::Writer Polygon(const RotatedBox& b) {
  pb::Writer poly;
  if (b.curve) {
    pb::Writer curved = CurveFields(*b.curve);
    curved.Message(4, Box(b));
    poly.Message(3, curved);
  } else {
    poly.Message(2, Box(b));
  }
  return poly;
}

RotatedBox ReadBox(std::string_view b);

Curve ReadCurve(std::string_view b, RotatedBox* rotated_bounding_box) {
  Curve c;
  pb::Reader r(b);
  while (r.Next()) {
    switch (r.field()) {
      case 1: {
        pb::Reader pts(r.bytes());
        while (pts.Next()) {
          if (pts.field() != 1) continue;
          std::pair<double, double> p{0, 0};
          pb::Reader pt(pts.bytes());
          while (pt.Next()) (pt.field() == 1 ? p.first : p.second) = pt.float64();
          c.points.push_back(p);
        }
        break;
      }
      case 2: c.thickness = r.float64(); break;
      case 3: c.top_to_bottom = r.varint() != 0; break;
      case 4: if (rotated_bounding_box) *rotated_bounding_box = ReadBox(r.bytes()); break;
    }
  }
  return c;
}

RotatedBox ReadBox(std::string_view b) {
  RotatedBox box;
  pb::Reader r(b);
  while (r.Next()) {
    switch (r.field()) {
      case 1: box.left = static_cast<int>(r.int64()); break;
      case 2: box.top = static_cast<int>(r.int64()); break;
      case 3: box.width = static_cast<int>(r.int64()); break;
      case 4: box.height = static_cast<int>(r.int64()); break;
      case 5: box.angle = r.float32(); break;
      case 6: box.curve = ReadCurve(r.bytes(), nullptr); break;  // ocr.photo.BoundingBox.curved_box
    }
  }
  return box;
}

void ReadDetection(std::string_view b, DetectionInfo* d) {
  pb::Reader r(b);
  while (r.Next()) {
    switch (r.field()) {
      case 1: d->detection_box = ReadBox(r.bytes()); break;
      case 4: d->detection_image_scale = static_cast<int>(r.int64()); break;
      case 9: d->average_raw_detections_height = r.float32(); break;
      case 10: d->variance_raw_detections_height = r.float32(); break;
      case 11: d->detection_nms_score = r.float32(); break;
      case 12: d->num_raw_boxes = static_cast<int>(r.int64()); break;
      case 13: d->sum_raw_boxes_scores = r.float32(); break;
      case 14: d->left_padding = r.float32(); break;
      case 15: d->right_padding = r.float32(); break;
      case 16: d->garbage_prob = r.float32(); break;
      case 17: d->fallback_upscaling = r.varint() != 0; break;
    }
  }
}

void ReadColors(std::string_view b, Colors* c) {
  pb::Reader r(b);
  auto point = [](std::string_view pb_bytes) {
    std::pair<int, int> p{0, 0};
    pb::Reader pr(pb_bytes);
    while (pr.Next()) (pr.field() == 1 ? p.first : p.second) = static_cast<int>(pr.int64());
    return p;
  };
  while (r.Next()) {
    switch (r.field()) {
      case 1: c->foreground_gray = static_cast<int>(r.int64()); break;
      case 2: c->background_gray = static_cast<int>(r.int64()); break;
      case 3: c->foreground_rgb = static_cast<int>(r.int64()); break;
      case 4: c->background_rgb = static_cast<int>(r.int64()); break;
      case 5: c->foreground_gray_threshold = static_cast<int>(r.int64()); break;
      case 6: c->foreground_samples.push_back(point(r.bytes())); break;
      case 7: c->background_samples.push_back(point(r.bytes())); break;
    }
  }
}

bool ReadEntity(std::string_view b, Entity* e) {
  pb::Reader r(b);
  e->parent = -1;
  while (r.Next()) {
    switch (r.field()) {
      case 1: e->parent = static_cast<int>(r.int64()); break;
      case 2: {
        pb::Reader poly(r.bytes());
        while (poly.Next())
          if (poly.field() == 2) {
            e->box = ReadBox(poly.bytes());
          } else if (poly.field() == 3) {
            RotatedBox box;
            Curve c = ReadCurve(poly.bytes(), &box);
            box.curve = std::move(c);
            e->box = box;
          }
        break;
      }
      case 4: e->type = static_cast<EntityType>(r.int64()); break;
      case 5: e->content_type = static_cast<int>(r.int64()); break;
      case 6: {
        pb::Reader o(r.bytes());
        while (o.Next()) {
          if (o.field() == 1) e->orientation = static_cast<int>(o.int64());
          if (o.field() == 2) e->writing_direction = static_cast<int>(o.int64());
          if (o.field() == 3) e->textline_order = static_cast<int>(o.int64());
        }
        break;
      }
      case 7: e->confidence = r.float32(); break;
      case 13: {
        pb::Reader mix(r.bytes());
        while (mix.Next()) {
          if (mix.field() != 1) continue;
          pb::Reader lang(mix.bytes());
          while (lang.Next())
            if (lang.field() == 1) e->languages.emplace_back(lang.bytes());
        }
        break;
      }
      case 14: e->detection_confidence = r.float32(); break;
      case 15: e->text = std::string(r.bytes()); break;
      case 16: {
        WordBreak wb;
        pb::Reader br(r.bytes());
        while (br.Next()) {
          if (br.field() == 1) wb.type = static_cast<int>(br.int64());
          if (br.field() == 2) wb.box = ReadBox(br.bytes());
        }
        e->word_break = wb;
        break;
      }
      case 17: {
        Colors c;
        ReadColors(r.bytes(), &c);
        e->colors = c;
        break;
      }
      case 22: e->type_confidence = r.float32(); break;
      case 23: e->recognizer_script_id = std::string(r.bytes()); break;
      case 27: e->direction_confidence = r.float32(); break;
      case 32: e->predicted_direction = static_cast<int>(r.int64()); break;
      case kPhotoOcrTextDetectionInfoExt: {
        DetectionInfo d;
        ReadDetection(r.bytes(), &d);
        e->detection = d;
        break;
      }
      case kAksaraInternalEntityExt: {
        pb::Reader ext(r.bytes());
        while (ext.Next()) {
          if (ext.field() != 3) continue;
          AksaraResult a;
          pb::Reader res(ext.bytes());
          while (res.Next()) {
            if (res.field() == 1) a.text = std::string(res.bytes());
            if (res.field() == 19) a.right_to_left = res.varint() != 0;
            if (res.field() == 20) a.language = std::string(res.bytes());
            if (res.field() == 21) a.confidence = res.float32();
            if (res.field() == 24) a.predicted_script = std::string(res.bytes());
          }
          e->aksara = a;
        }
        break;
      }
    }
  }
  return r.ok();
}

}  // namespace

bool ReadGoodocProto(std::string_view bytes, PageLayout* layout) {
  *layout = PageLayout();
  pb::Reader r(bytes);
  while (r.Next()) {
    switch (r.field()) {
      case 1: layout->width = static_cast<int>(r.int64()); break;
      case 2: layout->height = static_cast<int>(r.int64()); break;
      case 7: {
        Entity e;
        if (!ReadEntity(r.bytes(), &e)) return false;
        layout->entities.push_back(std::move(e));
        break;
      }
      case 8: {
        pb::Reader mix(r.bytes());
        if (!layout->language_mix) layout->language_mix.emplace();
        while (mix.Next()) {
          if (mix.field() != 1) continue;
          std::pair<std::string, float> lang{"", 0.0f};
          pb::Reader lr(mix.bytes());
          while (lr.Next()) {
            if (lr.field() == 1) lang.first = std::string(lr.bytes());
            if (lr.field() == 2) lang.second = lr.float32();
          }
          layout->language_mix->push_back(lang);
        }
        break;
      }
      case 9: layout->creator_id = std::string(r.bytes()); break;
    }
  }
  return r.ok();
}

std::string WriteGoodocProto(const PageLayout& layout) {
  pb::Writer w;
  w.Int(1, layout.width);
  w.Int(2, layout.height);
  pb::Writer image;
  image.Int(2, layout.width);
  image.Int(3, layout.height);
  image.Int(4, 1);
  w.Message(3, image);
  for (const Entity& e : layout.entities) w.Message(7, Entity_(e));
  if (layout.language_mix) {
    pb::Writer mix;
    for (const auto& [tag, weight] : *layout.language_mix) {
      pb::Writer lang;
      lang.Bytes(1, tag);
      lang.Float(2, weight);
      mix.Message(1, lang);
    }
    w.Message(8, mix);
  }
  if (layout.creator_id) w.Bytes(9, *layout.creator_id);
  return w.str();
}

}  // namespace screenai
