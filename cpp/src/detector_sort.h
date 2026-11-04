// SPDX-License-Identifier: MPL-2.0
// The detector's sorts in the original are libc++'s std::sort (introsort with pdqsort-style partitioning, LLVM
// 17+ algorithm, non-branchless path for lambda comparators). Several of them sort keys that can tie (equal link
// costs, equal projections, equal paragraph ids), where the resulting order of the tied elements changes the
// output. This is a self-contained port of that algorithm so the tie order is the original's on any standard
// library (verified against the original's own instantiations with trace/det_sort_check.py).
#pragma once

#include <cstddef>
#include <iterator>
#include <utility>

namespace screenai {
namespace detsort {

template <class It, class C>
bool Sort3(It x, It y, It z, C& c) {
  using std::swap;
  if (!c(*y, *x)) {
    if (!c(*z, *y)) return false;
    swap(*y, *z);
    if (c(*y, *x)) swap(*x, *y);
    return true;
  }
  if (c(*z, *y)) {
    swap(*x, *z);
    return true;
  }
  swap(*x, *y);
  if (c(*z, *y)) swap(*y, *z);
  return true;
}

template <class It, class C>
void Sort4(It x1, It x2, It x3, It x4, C& c) {
  using std::swap;
  Sort3(x1, x2, x3, c);
  if (c(*x4, *x3)) {
    swap(*x3, *x4);
    if (c(*x3, *x2)) {
      swap(*x2, *x3);
      if (c(*x2, *x1)) swap(*x1, *x2);
    }
  }
}

template <class It, class C>
void Sort5(It x1, It x2, It x3, It x4, It x5, C& c) {
  using std::swap;
  Sort4(x1, x2, x3, x4, c);
  if (c(*x5, *x4)) {
    swap(*x4, *x5);
    if (c(*x4, *x3)) {
      swap(*x3, *x4);
      if (c(*x3, *x2)) {
        swap(*x2, *x3);
        if (c(*x2, *x1)) swap(*x1, *x2);
      }
    }
  }
}

template <class It, class C>
void InsertionSort(It first, It last, C& c) {
  using V = typename std::iterator_traits<It>::value_type;
  if (first == last) return;
  for (It i = first + 1; i != last; ++i) {
    It j = i - 1;
    if (c(*i, *j)) {
      V t(std::move(*i));
      It k = j;
      j = i;
      do {
        *j = std::move(*k);
        j = k;
      } while (j != first && c(t, *--k));
      *j = std::move(t);
    }
  }
}

template <class It, class C>
void InsertionSortUnguarded(It first, It last, C& c) {
  using V = typename std::iterator_traits<It>::value_type;
  if (first == last) return;
  for (It i = first + 1; i != last; ++i) {
    It j = i - 1;
    if (c(*i, *j)) {
      V t(std::move(*i));
      It k = j;
      j = i;
      do {
        *j = std::move(*k);
        j = k;
      } while (c(t, *--k));
      *j = std::move(t);
    }
  }
}

template <class It, class C>
bool InsertionSortIncomplete(It first, It last, C& c) {
  using V = typename std::iterator_traits<It>::value_type;
  using std::swap;
  switch (last - first) {
    case 0:
    case 1:
      return true;
    case 2:
      if (c(*--last, *first)) swap(*first, *last);
      return true;
    case 3:
      Sort3(first, first + 1, --last, c);
      return true;
    case 4:
      Sort4(first, first + 1, first + 2, --last, c);
      return true;
    case 5:
      Sort5(first, first + 1, first + 2, first + 3, --last, c);
      return true;
  }
  It j = first + 2;
  Sort3(first, first + 1, j, c);
  const unsigned kLimit = 8;
  unsigned count = 0;
  for (It i = j + 1; i != last; ++i) {
    if (c(*i, *j)) {
      V t(std::move(*i));
      It k = j;
      j = i;
      do {
        *j = std::move(*k);
        j = k;
      } while (j != first && c(t, *--k));
      *j = std::move(t);
      if (++count == kLimit) return ++i == last;
    }
    j = i;
  }
  return true;
}

template <class It, class C>
std::pair<It, bool> PartitionWithEqualsOnRight(It first, It last, C& c) {
  using V = typename std::iterator_traits<It>::value_type;
  using std::swap;
  const It begin = first;
  V pivot(std::move(*first));
  do ++first;
  while (c(*first, pivot));
  if (begin == first - 1) {
    while (first < last && !c(*--last, pivot)) {
    }
  } else {
    do --last;
    while (!c(*last, pivot));
  }
  const bool already_partitioned = first >= last;
  while (first < last) {
    swap(*first, *last);
    do ++first;
    while (c(*first, pivot));
    do --last;
    while (!c(*last, pivot));
  }
  It pivot_pos = first - 1;
  if (begin != pivot_pos) *begin = std::move(*pivot_pos);
  *pivot_pos = std::move(pivot);
  return {pivot_pos, already_partitioned};
}

template <class It, class C>
It PartitionWithEqualsOnLeft(It first, It last, C& c) {
  using V = typename std::iterator_traits<It>::value_type;
  using std::swap;
  const It begin = first;
  V pivot(std::move(*first));
  if (c(pivot, *(last - 1))) {
    do ++first;
    while (!c(pivot, *first));
  } else {
    while (++first < last && !c(pivot, *first)) {
    }
  }
  if (first < last) {
    do --last;
    while (c(pivot, *last));
  }
  while (first < last) {
    swap(*first, *last);
    do ++first;
    while (!c(pivot, *first));
    do --last;
    while (c(pivot, *last));
  }
  It pivot_pos = first - 1;
  if (begin != pivot_pos) *begin = std::move(*pivot_pos);
  *pivot_pos = std::move(pivot);
  return first;
}

template <class It, class C>
void SiftDown(It first, C& c, std::ptrdiff_t len, It start) {
  using V = typename std::iterator_traits<It>::value_type;
  std::ptrdiff_t child = start - first;
  if (len < 2 || (len - 2) / 2 < child) return;
  child = 2 * child + 1;
  It child_i = first + child;
  if (child + 1 < len && c(*child_i, *(child_i + 1))) ++child_i, ++child;
  if (c(*child_i, *start)) return;
  V top(std::move(*start));
  do {
    *start = std::move(*child_i);
    start = child_i;
    if ((len - 2) / 2 < child) break;
    child = 2 * child + 1;
    child_i = first + child;
    if (child + 1 < len && c(*child_i, *(child_i + 1))) ++child_i, ++child;
  } while (!c(*child_i, top));
  *start = std::move(top);
}

template <class It, class C>
It FloydSiftDown(It first, C& c, std::ptrdiff_t len) {
  It hole = first, child_i = first;
  std::ptrdiff_t child = 0;
  while (true) {
    child_i += child + 1;
    child = 2 * child + 1;
    if (child + 1 < len && c(*child_i, *(child_i + 1))) ++child_i, ++child;
    *hole = std::move(*child_i);
    hole = child_i;
    if (child > (len - 2) / 2) return hole;
  }
}

template <class It, class C>
void SiftUp(It first, It last, C& c, std::ptrdiff_t len) {
  using V = typename std::iterator_traits<It>::value_type;
  if (len > 1) {
    len = (len - 2) / 2;
    It ptr = first + len;
    if (c(*ptr, *--last)) {
      V t(std::move(*last));
      do {
        *last = std::move(*ptr);
        last = ptr;
        if (len == 0) break;
        len = (len - 1) / 2;
        ptr = first + len;
      } while (c(*ptr, t));
      *last = std::move(t);
    }
  }
}

template <class It, class C>
void HeapSort(It first, It last, C& c) {  // __partial_sort(first, last, last)
  using V = typename std::iterator_traits<It>::value_type;
  const std::ptrdiff_t n = last - first;
  if (n == 0) return;
  if (n > 1)
    for (std::ptrdiff_t start = (n - 2) / 2; start >= 0; --start) SiftDown(first, c, n, first + start);
  for (std::ptrdiff_t m = n; m > 1; --last, --m) {
    V top(std::move(*first));
    It hole = FloydSiftDown(first, c, m);
    It back = last - 1;
    if (hole == back) {
      *hole = std::move(top);
    } else {
      *hole = std::move(*back);
      ++hole;
      *back = std::move(top);
      SiftUp(first, hole, c, hole - first);
    }
  }
}

template <class It, class C>
void Introsort(It first, It last, C& c, std::ptrdiff_t depth, bool leftmost) {
  using std::swap;
  constexpr std::ptrdiff_t kLimit = 24, kNintherThreshold = 128;
  while (true) {
    const std::ptrdiff_t len = last - first;
    switch (len) {
      case 0:
      case 1:
        return;
      case 2:
        if (c(*--last, *first)) swap(*first, *last);
        return;
      case 3:
        Sort3(first, first + 1, --last, c);
        return;
      case 4:
        Sort4(first, first + 1, first + 2, --last, c);
        return;
      case 5:
        Sort5(first, first + 1, first + 2, first + 3, --last, c);
        return;
    }
    if (len < kLimit) {
      if (leftmost)
        InsertionSort(first, last, c);
      else
        InsertionSortUnguarded(first, last, c);
      return;
    }
    if (depth == 0) {
      HeapSort(first, last, c);
      return;
    }
    --depth;
    const std::ptrdiff_t half = len / 2;
    if (len > kNintherThreshold) {
      Sort3(first, first + half, last - 1, c);
      Sort3(first + 1, first + (half - 1), last - 2, c);
      Sort3(first + 2, first + (half + 1), last - 3, c);
      Sort3(first + (half - 1), first + half, first + (half + 1), c);
      swap(*first, *(first + half));
    } else {
      Sort3(first + half, first, last - 1, c);
    }
    if (!leftmost && !c(*(first - 1), *first)) {
      first = PartitionWithEqualsOnLeft(first, last, c);
      continue;
    }
    auto ret = PartitionWithEqualsOnRight(first, last, c);
    It i = ret.first;
    if (ret.second) {
      const bool fs = InsertionSortIncomplete(first, i, c);
      if (InsertionSortIncomplete(i + 1, last, c)) {
        if (fs) return;
        last = i;
        continue;
      }
      if (fs) {
        first = ++i;
        continue;
      }
    }
    Introsort(first, i, c, depth, leftmost);
    leftmost = false;
    first = ++i;
  }
}

// std::sort(first, last, comp) as libc++ implements it.
template <class It, class C>
void Sort(It first, It last, C comp) {
  std::ptrdiff_t n = last - first, log2 = 0;
  while (n > 1) n >>= 1, log2++;
  Introsort(first, last, comp, 2 * log2, true);
}

}  // namespace detsort
}  // namespace screenai
