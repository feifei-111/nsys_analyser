from .parser.parse_json import FilterResult


class NodeFilter:

    @classmethod
    def range_filter(cls, start, end):
        start = start
        end = end

        def _filter(node):
            if node.start < start:
                return FilterResult.DROP
            elif node.start < end:
                return FilterResult.SAVE
            else:
                return FilterResult.BREAK

        return _filter

    @classmethod
    def text_filter(cls, text):
        range_filter = None

        def _filter(node):
            nonlocal range_filter
            if range_filter is not None:
                return range_filter(node)
            if node.text == text:
                range_filter = NodeFilter.range_filter(node.start, node.end)
                return FilterResult.SAVE
            else:
                return FilterResult.DROP

        return _filter
