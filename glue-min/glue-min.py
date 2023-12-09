#!/usr/bin/env python
import re
import os
import codecs
import sys
import argparse
import textwrap
import contextlib # Helpers
import copy # Core, Algorithms/Square
import hashlib # Core
from io import StringIO, BytesIO # Core
import configparser

from PIL import __version__ as PILVersion
from PIL import Image as PILImage # Core
from PIL import PngImagePlugin # Formats/Img

from jinja2 import Template # Formats/Base

# START Version
__version__ = '0.13'
# END Version

# START Helpers
def round_up(value):
    int_value = int(value)
    diff = 1 if value > 0 else -1
    return int_value + diff if value != int_value else int_value


def nearest_fraction(value):
    return '{}/100'.format(int(float(value) * 100))


class _Missing(object):
    """ Missing object necessary for cached_property"""
    def __repr__(self):
        return 'no value'

    def __reduce__(self):
        return '_missing'

_missing = _Missing()


class cached_property(object):
    """
    Decorator inspired/copied from mitsuhiko/werkzeug.

    A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value"""

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


@contextlib.contextmanager
def redirect_stdout(stream=None):
    stream = stream or StringIO()
    sys.stdout = stream
    yield
    sys.stdout = sys.__stdout__
# END Helpers

# START Algorithms
class SquareAlgorithmNode(object):

    def __init__(self, x=0, y=0, width=0, height=0, used=False,
                 down=None, right=None):
        """Node constructor.

        :param x: X coordinate.
        :param y: Y coordinate.
        :param width: Image width.
        :param height: Image height.
        :param used: Flag to determine if the node is used.
        :param down: Down :class:`~Node`.
        :param right Right :class:`~Node`.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.used = used
        self.right = right
        self.down = down

    def find(self, node, width, height):
        """Find a node to allocate this image size (width, height).

        :param node: Node to search in.
        :param width: Pixels to grow down (width).
        :param height: Pixels to grow down (height).
        """
        if node.used:
            return self.find(node.right, width, height) or self.find(node.down, width, height)
        elif node.width >= width and node.height >= height:
            return node
        return None

    def grow(self, width, height):
        """ Grow the canvas to the most appropriate direction.

        :param width: Pixels to grow down (width).
        :param height: Pixels to grow down (height).
        """
        can_grow_d = width <= self.width
        can_grow_r = height <= self.height

        should_grow_r = can_grow_r and self.height >= (self.width + width)
        should_grow_d = can_grow_d and self.width >= (self.height + height)

        if should_grow_r:
            return self.grow_right(width, height)
        elif should_grow_d:
            return self.grow_down(width, height)
        elif can_grow_r:
            return self.grow_right(width, height)
        elif can_grow_d:
            return self.grow_down(width, height)

        return None

    def grow_right(self, width, height):
        """Grow the canvas to the right.

        :param width: Pixels to grow down (width).
        :param height: Pixels to grow down (height).
        """
        old_self = copy.copy(self)
        self.used = True
        self.x = self.y = 0
        self.width += width
        self.down = old_self
        self.right = SquareAlgorithmNode(x=old_self.width,
                                         y=0,
                                         width=width,
                                         height=self.height)

        node = self.find(self, width, height)
        if node:
            return self.split(node, width, height)
        return None

    def grow_down(self, width, height):
        """Grow the canvas down.

        :param width: Pixels to grow down (width).
        :param height: Pixels to grow down (height).
        """
        old_self = copy.copy(self)
        self.used = True
        self.x = self.y = 0
        self.height += height
        self.right = old_self
        self.down = SquareAlgorithmNode(x=0,
                                        y=old_self.height,
                                        width=self.width,
                                        height=height)

        node = self.find(self, width, height)
        if node:
            return self.split(node, width, height)
        return None

    def split(self, node, width, height):
        """Split the node to allocate a new one of this size.

        :param node: Node to be splitted.
        :param width: New node width.
        :param height: New node height.
        """
        node.used = True
        node.down = SquareAlgorithmNode(x=node.x,
                                        y=node.y + height,
                                        width=node.width,
                                        height=node.height - height)
        node.right = SquareAlgorithmNode(x=node.x + width,
                                         y=node.y,
                                         width=node.width - width,
                                         height=height)
        return node

class SquareAlgorithm(object):

    def process(self, sprite):

        root = SquareAlgorithmNode(width=sprite.images[0].absolute_width,
                                   height=sprite.images[0].absolute_height)

        # Loot all over the images creating a binary tree
        for image in sprite.images:
            node = root.find(root, image.absolute_width, image.absolute_height)
            if node:  # Use this node
                node = root.split(node, image.absolute_width, image.absolute_height)
            else:  # Grow the canvas
                node = root.grow(image.absolute_width, image.absolute_height)

            image.x = node.x
            image.y = node.y

# END Algorithms

# START Managers/Base
class BaseManager(object):

    def __init__(self, *args, **kwargs):
        self.config = kwargs
        self.sprites = []

    def process(self):
        self.find_sprites()
        self.validate()
        self.save()

    def add_sprite(self, path):
        """Create a new Sprite using this path and name and append it to the
        sprites list.

        :param path: Sprite path.
        :param name: Sprite name.
        """
        sprite = Sprite(path=path, config=self.config)
        self.sprites.append(sprite)

    def find_sprites(self):
        raise NotImplementedError

    def validate(self):
        """Validate all sprites inside this manager."""

        for sprite in self.sprites:
            sprite.validate()

    def save(self):
        """Save all sprites inside this manager."""

        for format_name in self.config['enabled_formats']:
            format_cls = formats[format_name]
            for sprite in self.sprites:
                format = format_cls(sprite=sprite)
                format.validate()
                if format.needs_rebuild() or sprite.config['force']:
                    print("Format '{0}' for sprite '{1}' needs rebuild...".format(format_name, sprite.name))
                    format.build()
                else:
                    print("Format '{0}'' for sprite '{1}' already exists...".format(format_name, sprite.name))
# END Managers/Base

# START Core
class ConfigurableFromFile(object):

    def _get_config_from_file(self, filename, section):
        """Return, as a dictionary, all the available configuration inside the
        sprite configuration file on this sprite path."""

        def clean(value):
            return {'true': True, 'false': False}.get(value.lower(), value)

        config = configparser.RawConfigParser()
        config.read(os.path.join(self.config_path, filename))
        try:
            keys = config.options(section)
        except configparser.NoSectionError:
            return {}
        return dict([[k, clean(config.get(section, k))] for k in keys])


class Image(ConfigurableFromFile):

    def __init__(self, path, config):
        self.path = path
        self.filename = os.path.basename(path)
        self.dirname = self.config_path = os.path.dirname(path)

        self.config = copy.deepcopy(config)
        self.config.update(self._get_config_from_file('sprite.conf', self.filename))

        self.x = self.y = None
        self.original_width = self.original_height = 0

        with open(self.path, "rb") as img:
            self._image_data = img.read()

        print("\t{0} added to sprite".format(self.filename))

    @cached_property
    def image(self):
        """Return a Pil representation of this image """

        imageio = BytesIO(self._image_data)

        try:
            source_image = PILImage.open(imageio)
            img = PILImage.new('RGBA', source_image.size, (0, 0, 0, 0))

            if source_image.mode == 'L':
                alpha = source_image.split()[0]
                transparency = source_image.info.get('transparency')
                mask = PILImage.eval(alpha, lambda a: 0 if a == transparency else 255)
                img.paste(source_image, (0, 0), mask=mask)
            else:
                img.paste(source_image, (0, 0))
        except IOError as e:
            raise PILUnavailableError(e.args[0].split()[1])
        finally:
            imageio.close()

        self.original_width, self.original_height = img.size

        # Crop the image searching for the smallest possible bounding box
        # without losing any non-transparent pixel.
        # This crop is only used if the crop flag is set in the config.
        if self.config['crop']:
            img = img.crop(img.split()[-1].getbbox())
        return img

    @property
    def width(self):
        """Return Image width"""
        return self.image.size[0]

    @property
    def height(self):
        """Return Image height"""
        return self.image.size[1]

    @property
    def padding(self):
        """Return a 4-elements list with the desired padding."""
        return self._generate_spacing_info(self.config['padding'])

    @property
    def margin(self):
        """Return a 4-elements list with the desired marging."""
        return self._generate_spacing_info(self.config['margin'])

    def _generate_spacing_info(self, data):

        data = data.split(',' if ',' in data else ' ')

        if len(data) == 4:
            data = data
        elif len(data) == 3:
            data = data + [data[1]]
        elif len(data) == 2:
            data = data * 2
        elif len(data) == 1:
            data = data * 4
        else:
            data = [0] * 4

        return list(map(int, data))

    @cached_property
    def horizontal_spacing(self):
        return self.padding[1] + self.padding[3] + self.margin[1] + self.margin[3]

    @cached_property
    def vertical_spacing(self):
        return self.padding[0] + self.padding[2] + self.margin[0] + self.margin[2]

    @property
    def absolute_width(self):
        return round_up(self.width + self.horizontal_spacing * max(self.config['ratios']))

    @property
    def absolute_height(self):
        return round_up(self.height + self.vertical_spacing * max(self.config['ratios']))

    def __lt__(self, img):
        return max(self.absolute_width, self.absolute_height) <= max(img.absolute_width, img.absolute_height)


class Sprite(ConfigurableFromFile):

    config_filename = 'sprite.conf'
    config_section = 'sprite'
    valid_extensions = ['png', 'jpg', 'jpeg', 'gif']

    def __init__(self, path, config, name=None):
        self.path = self.config_path = path
        self.config = copy.deepcopy(config)
        self.config.update(self._get_config_from_file('sprite.conf', 'sprite'))
        self.name = name or self.config.get('name', os.path.basename(path))

        # Setup ratios
        ratios = self.config['ratios'].split(',')
        ratios = set([float(r.strip()) for r in ratios if r.strip()])

        # Always add 1.0 as a required ratio
        ratios.add(1.0)

        # Create a sorted list of ratios
        self.ratios = sorted(ratios)
        self.max_ratio = max(self.ratios)
        self.config['ratios'] = self.ratios

        # Discover images inside this sprite
        self.images = self._locate_images()

        img_format = ImageFormat(sprite=self)
        for ratio in ratios:
            ratio_output_key = 'ratio_{0}_output'.format(ratio)
            if ratio_output_key not in self.config:
                self.config[ratio_output_key] = img_format.output_path(ratio)

        print("Processing '{0}':".format(self.name))

        # Generate sprite map
        self.process()

    def process(self):
        algorithm_cls = algorithms[self.config['algorithm']]
        algorithm = algorithm_cls()
        algorithm.process(self)

    def validate(self):
        pass

    @cached_property
    def hash(self):
        """ Return a hash of this sprite. In order to detect any change on
        the source images  it use the data, order and path of each image.
        In the same way it use this sprite settings as part of the hash.
        """
        hash_list = []
        for image in self.images:
            hash_list.append(os.path.relpath(image.path))
            hash_list.append(image._image_data)

        for key, value in self.config.items():
            hash_list.append(key)
            hash_list.append(value)

        return hashlib.sha1(''.join(map(str, hash_list)).encode('utf-8')).hexdigest()[:10]

    @cached_property
    def canvas_size(self):
        """Return the width and height for this sprite canvas"""
        width = height = 0
        for image in self.images:
            x = image.x + image.absolute_width
            y = image.y + image.absolute_height
            if width < x:
                width = x
            if height < y:
                height = y
        return round_up(width), round_up(height)

    def sprite_path(self, ratio=1.0):
        return self.config['ratio_{0}_output'.format(ratio)]

    def _locate_images(self):
        """Return all valid images within a folder.

        All files with a extension not included in
        (png, jpg, jpeg and gif) or beginning with '.' will be ignored.

        If the folder doesn't contain any valid image it will raise
        :class:`~SourceImagesNotFoundError`

        The list of images will be ordered using the desired ordering
        algorithm. The default is 'maxside'.
        """
        extensions = '|'.join(self.valid_extensions)
        extension_re = re.compile('.+\\.(%s)$' % extensions, re.IGNORECASE)
        files = sorted(os.listdir(self.path))

        images = []
        for root, dirs, files in os.walk(self.path, followlinks=self.config['follow_links']):
            for filename in sorted(files):
                if not filename.startswith('.') and extension_re.match(filename):
                    images.append(Image(path=os.path.join(root, filename), config=self.config))
            if not self.config['recursive']:
                break

        if not images:
            raise SourceImagesNotFoundError(self.path)

        images = sorted(images, reverse=self.config['algorithm_ordering'][0] != '-')

        return images
# END Core

# START Managers/Simple
class SimpleManager(BaseManager):
    """Process a single folder and create one sprite. It works the
    same way as :class:`~ProjectSpriteManager`, but only for one folder.

    This is the default manager.
    """

    def find_sprites(self):
        self.add_sprite(path=self.config['source'])
# END Managers/Simple

# START Formats/Base
class BaseFormat(object):

    extension = None
    build_per_ratio = False

    def __init__(self, sprite):
        self.sprite = sprite

    def output_dir(self, *args, **kwargs):
        return self.sprite.config['{0}_dir'.format(self.format_label)]

    def output_filename(self, ratio=None, *args, **kwargs):
        if self.build_per_ratio:
            if ratio is None:
                raise AttributeError("Format {0} output_filename requires a ratio.".format(self.__class__))
            ratio_suffix = '@%.1fx' % ratio if int(ratio) != ratio else '@%ix' % ratio
            if ratio_suffix == '@1x':
                ratio_suffix = ''
            return '{0}{1}'.format(self.sprite.name, ratio_suffix)
        return self.sprite.name

    def output_path(self, *args, **kwargs):
        return os.path.join(self.output_dir(*args, **kwargs), '{0}.{1}'.format(self.output_filename(*args, **kwargs), self.extension))

    def build(self):
        if self.build_per_ratio:
            for ratio in self.sprite.config['ratios']:
                self.save(ratio=ratio)
        else:
            self.save()

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def needs_rebuild(self):
        return True

    def validate(self):
        pass

    @property
    def format_label(self):
        return dict((v,k) for k, v in formats.items())[self.__class__]

    @classmethod
    def populate_argument_parser(cls, parser):
        pass

    @classmethod
    def apply_parser_contraints(cls, parser, options):
        pass

    def fix_windows_path(self, path):
        if os.name == 'nt':
            path = path.replace('\\', '/')
        return path


class BaseTextFormat(BaseFormat):

    def get_context(self, *args, **kwargs):
        sprite_path = os.path.relpath(self.sprite.sprite_path(), self.output_dir())
        sprite_path = self.fix_windows_path(sprite_path)
        context = {'version': __version__,
                   'hash': self.sprite.hash,
                   'name': self.sprite.name,
                   'sprite_path': sprite_path,
                   'sprite_filename': os.path.basename(sprite_path),
                   'width': round_up(self.sprite.canvas_size[0] / self.sprite.max_ratio),
                   'height': round_up(self.sprite.canvas_size[1] / self.sprite.max_ratio),
                   'images': [],
                   'ratios': {}}

        for i, img in enumerate(self.sprite.images):
            base_x = img.x * -1 - img.margin[3] * self.sprite.max_ratio
            base_y = img.y * -1 - img.margin[0] * self.sprite.max_ratio
            base_abs_x = img.x + img.margin[3] * self.sprite.max_ratio
            base_abs_y = img.y + img.margin[0] * self.sprite.max_ratio

            image = dict(filename=img.filename,
                         last=i == len(self.sprite.images) - 1,
                         x=round_up(base_x / self.sprite.max_ratio),
                         y=round_up(base_y / self.sprite.max_ratio),
                         abs_x=round_up(base_abs_x / self.sprite.max_ratio),
                         abs_y=round_up(base_abs_y / self.sprite.max_ratio),
                         height=round_up((img.height / self.sprite.max_ratio) + img.padding[0] + img.padding[2]),
                         width=round_up((img.width / self.sprite.max_ratio) + img.padding[1] + img.padding[3]),
                         original_width=img.original_width,
                         original_height=img.original_height,
                         ratios={})

            for r in self.sprite.ratios:
                image['ratios'][r] = dict(filename=img.filename,
                                          last=i == len(self.sprite.images) - 1,
                                          x=round_up(base_x / self.sprite.max_ratio * r),
                                          y=round_up(base_y / self.sprite.max_ratio * r),
                                          abs_x=round_up(base_abs_x / self.sprite.max_ratio * r),
                                          abs_y=round_up(base_abs_y / self.sprite.max_ratio * r),
                                          height=round_up((img.height + img.padding[0] + img.padding[2]) / self.sprite.max_ratio * r),
                                          width=round_up((img.width + img.padding[1] + img.padding[3]) / self.sprite.max_ratio * r))

            context['images'].append(image)

        # Ratios
        for r in self.sprite.ratios:
            ratio_sprite_path = os.path.relpath(self.sprite.sprite_path(ratio=r), self.output_dir())
            ratio_sprite_path = self.fix_windows_path(ratio_sprite_path)
            context['ratios'][r] = dict(ratio=r,
                                        fraction=nearest_fraction(r),
                                        sprite_path=ratio_sprite_path,
                                        sprite_filename=os.path.basename(ratio_sprite_path),
                                        width=round_up(self.sprite.canvas_size[0] / self.sprite.max_ratio * r),
                                        height=round_up(self.sprite.canvas_size[1] / self.sprite.max_ratio * r))

        return context

    def render(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        # Create the destination directory if required
        if not os.path.exists(self.output_dir(*args, **kwargs)):
            os.makedirs(self.output_dir(*args, **kwargs))

        with codecs.open(self.output_path(*args, **kwargs), 'w', 'utf-8') as f:
            f.write(self.render(*args, **kwargs))


class JinjaTextFormat(BaseTextFormat):

    template = ''

    def render(self, *args, **kwargs):
        context = self.get_context(*args, **kwargs)
        template = self.template
        custom_template_config = '{0}_template'.format(self.format_label)
        if self.sprite.config.get(custom_template_config):
            with open(self.sprite.config[custom_template_config]) as f:
                template = f.read()
        return Template(textwrap.dedent(template).strip()).render(**context)
# END Formats/Base

# START Formats/IMG
class ImageFormat(BaseFormat):

    build_per_ratio = True
    extension = 'png'

    @classmethod
    def populate_argument_parser(cls, parser):
        parser.add_argument("--img", dest="img_dir", type=str, default=True, metavar='DIR', help="Output directory for img files")
        parser.add_argument("--no-img", dest="generate_image", action="store_false", default=True, help="Don't generate IMG files.")
        parser.add_argument("-c", "--crop", dest="crop", action='store_true', default=False, help="Crop images removing unnecessary transparent margins")
        parser.add_argument("-p", "--padding", dest="padding", type=str, default='0', help="Force this padding in all images")
        parser.add_argument("--margin", dest="margin", type=str, default='0', help="Force this margin in all images")
        parser.add_argument("--png8", action="store_true", dest="png8", default=False, help="The output image format will be png8 instead of png32")
        parser.add_argument("--ratios", dest="ratios", type=str, default='1', help="Create sprites based on these ratios")
        parser.add_argument("--retina", dest="ratios", default=False, action='store_const', const='2,1', help="Shortcut for --ratios=2,1")

    def output_filename(self, *args, **kwargs):
        filename = super(ImageFormat, self).output_filename(*args, **kwargs)
        if self.sprite.config['css_cachebuster_filename'] or self.sprite.config['css_cachebuster_only_sprites']:
            return '{0}_{1}'.format(filename, self.sprite.hash)
        return filename

    def needs_rebuild(self):
        for ratio in self.sprite.config['ratios']:
            image_path = self.output_path(ratio)
            try:
                existing = PILImage.open(image_path)
                assert existing.info['Software'] == 'glue-%s' % __version__
                assert existing.info['Comment'] == self.sprite.hash
                continue
            except Exception:
                return True
        return False

    @cached_property
    def _raw_canvas(self):
        # Create the sprite canvas
        width, height = self.sprite.canvas_size
        canvas = PILImage.new('RGBA', (width, height), (0, 0, 0, 0))

        # Paste the images inside the canvas
        for image in self.sprite.images:
            canvas.paste(image.image,
                (round_up(image.x + (image.padding[3] + image.margin[3]) * self.sprite.max_ratio),
                 round_up(image.y + (image.padding[0] + image.margin[0]) * self.sprite.max_ratio)))

        meta = PngImagePlugin.PngInfo()
        meta.add_text('Software', 'glue-%s' % __version__)
        meta.add_text('Comment', self.sprite.hash)

        # Customize how the png is going to be saved
        kwargs = dict(optimize=False, pnginfo=meta)

        if self.sprite.config['png8']:
            # Get the alpha band
            alpha = canvas.split()[-1]
            canvas = canvas.convert('RGB'
                        ).convert('P',
                                  palette=PILImage.ADAPTIVE,
                                  colors=255)

            # Set all pixel values below 128 to 255, and the rest to 0
            mask = PILImage.eval(alpha, lambda a: 255 if a <= 128 else 0)

            # Paste the color of index 255 and use alpha as a mask
            canvas.paste(255, mask)
            kwargs.update({'transparency': 255})
        return canvas, kwargs

    def save(self, ratio):
        width, height = self.sprite.canvas_size
        canvas, kwargs = self._raw_canvas

        # Loop all over the ratios and save one image for each one
        for ratio in self.sprite.config['ratios']:

            # Create the destination directory if required
            if not os.path.exists(self.output_dir(ratio=ratio)):
                os.makedirs(self.output_dir(ratio=ratio))

            image_path = self.output_path(ratio=ratio)

            # If this canvas isn't the biggest one scale it using the ratio
            if self.sprite.max_ratio != ratio:

                reduced_canvas = canvas.resize(
                                    (round_up((width / self.sprite.max_ratio) * ratio),
                                     round_up((height / self.sprite.max_ratio) * ratio)),
                                     PILImage.Resampling.LANCZOS) # Changed PILImage.ANTIALIAS to PILImage.Resampling.LANCZOS per DeprecationWarning
                reduced_canvas.save(image_path, **kwargs)
                # TODO: Use Imagemagick if it's available
            else:
                canvas.save(image_path, **kwargs)
# END Formats/IMG

# START Formats/CSS
class CssFormat(JinjaTextFormat):

    extension = 'css'
    camelcase_separator = 'camelcase'
    css_pseudo_classes = set(['link', 'visited', 'active', 'hover', 'focus',
                              'first-letter', 'first-line', 'first-child',
                              'before', 'after'])

    template = u"""
        /* glue: {{ version }} hash: {{ hash }} */
        {% for image in images %}.{{ image.label }}{{ image.pseudo }}{%- if not image.last %},{{"\n"}}{%- endif %}{%- endfor %} {
            background-image: url('{{ sprite_path }}');
            background-repeat: no-repeat;
        }
        {% for image in images %}
        .{{ image.label }}{{ image.pseudo }} {
            background-position: {{ image.x ~ ('px' if image.x) }} {{ image.y ~ ('px' if image.y) }};
            width: {{ image.width }}px;
            height: {{ image.height }}px;
        }
        {% endfor %}{% for r, ratio in ratios.items() %}
        @media screen and (-webkit-min-device-pixel-ratio: {{ ratio.ratio }}), screen and (min--moz-device-pixel-ratio: {{ ratio.ratio }}), screen and (-o-min-device-pixel-ratio: {{ ratio.fraction }}), screen and (min-device-pixel-ratio: {{ ratio.ratio }}), screen and (min-resolution: {{ ratio.ratio }}dppx) {
            {% for image in images %}.{{ image.label }}{{ image.pseudo }}{% if not image.last %},{{"\n"}}    {% endif %}{% endfor %} {
                background-image: url('{{ ratio.sprite_path }}');
                -webkit-background-size: {{ width }}px {{ height }}px;
                -moz-background-size: {{ width }}px {{ height }}px;
                background-size: {{ width }}px {{ height }}px;
            }
        }
        {% endfor %}
        """

    @classmethod
    def populate_argument_parser(cls, parser):
        parser.add_argument("--css", dest="css_dir", nargs='?', const=True, default=False, metavar='DIR', help="Generate CSS files and optionally where")
        parser.add_argument("--namespace", dest="css_namespace", type=str, default='sprite', help="Namespace for all css classes (default: sprite)")
        parser.add_argument("--sprite-namespace", dest="css_sprite_namespace", type=str, default='{sprite_name}', help="Namespace for all sprites (default: {sprite_name})")
        parser.add_argument("-u", "--url", dest="css_url", type=str, default='', help="Prepend this string to the sprites path")
        parser.add_argument("--cachebuster", dest="css_cachebuster", default=False, action='store_true', help=("Use the sprite's sha1 first 6 characters as a queryarg everytime that file is referred from the css"))
        parser.add_argument("--cachebuster-filename", dest="css_cachebuster_filename", default=False, action='store_true', help=("Append the sprite's sha first 6 characters to the output filename"))
        parser.add_argument("--cachebuster-filename-only-sprites", dest="css_cachebuster_only_sprites", default=False, action='store_true', help=("Only apply cachebuster to sprite images."))
        parser.add_argument("--separator", dest="css_separator", type=str, default='-', metavar='SEPARATOR', help=("Customize the separator used to join CSS class names. If you want to use camelCase use 'camelcase' as separator."))
        parser.add_argument("--pseudo-class-separator", dest="css_pseudo_class_separator", type=str, default='__', metavar='SEPARATOR', help=("Customize the separator glue will use in order to determine the pseudo classes included into filenames."))
        parser.add_argument("--css-template", dest="css_template", default=None, metavar='DIR', help="Template to use to generate the CSS output.")
        parser.add_argument("--no-css", dest="generate_css", action="store_false", default=True, help="Don't generate CSS files.")

    @classmethod
    def apply_parser_contraints(cls, parser, options):
        cachebusters = (options.css_cachebuster, options.css_cachebuster_filename, options.css_cachebuster_only_sprites)
        if sum(cachebusters) > 1:
            parser.error("You can't use --cachebuster, --cachebuster-filename or --cachebuster-filename-only-sprites at the same time.")

    def needs_rebuild(self):
        hash_line = '/* glue: %s hash: %s */\n' % (__version__, self.sprite.hash)
        try:
            with codecs.open(self.output_path(), 'r', 'utf-8') as existing_css:
                first_line = existing_css.readline()
                assert first_line == hash_line
        except Exception:
            return True
        return False

    def validate(self):
        class_names = [':'.join(self.generate_css_name(i.filename)) for i in self.sprite.images]
        if len(set(class_names)) != len(self.sprite.images):
            dup = [i for i in self.sprite.images if class_names.count(':'.join(self.generate_css_name(i.filename))) > 1]
            duptext = '\n'.join(['\t{0} => .{1}'.format(os.path.relpath(d.path), ':'.join(self.generate_css_name(d.filename))) for d in dup])
            raise ValidationError("Error: Some images will have the same class name:\n{0}".format(duptext))
        return True

    def output_filename(self, *args, **kwargs):
        filename = super(CssFormat, self).output_filename(*args, **kwargs)
        if self.sprite.config['css_cachebuster_filename']:
            return '{0}_{1}'.format(filename, self.sprite.hash)
        return filename

    def get_context(self, *args, **kwargs):

        context = super(CssFormat, self).get_context(*args, **kwargs)

        # Generate css labels
        for image in context['images']:
            image['label'], image['pseudo'] = self.generate_css_name(image['filename'])

        if self.sprite.config['css_url']:
            context['sprite_path'] = '{0}{1}'.format(self.sprite.config['css_url'], context['sprite_filename'])

            for r, ratio in context['ratios'].items():
                ratio['sprite_path'] = '{0}{1}'.format(self.sprite.config['css_url'], ratio['sprite_filename'])

        # Add cachebuster if required
        if self.sprite.config['css_cachebuster']:

            def apply_cachebuster(path):
                return "%s?%s" % (path, self.sprite.hash)

            context['sprite_path'] = apply_cachebuster(context['sprite_path'])

            for r, ratio in context['ratios'].items():
                ratio['sprite_path'] = apply_cachebuster(ratio['sprite_path'])

        return context

    def generate_css_name(self, filename):
        filename = filename.rsplit('.', 1)[0]
        separator = self.sprite.config['css_separator']
        namespace = [re.sub(r'[^\w\-_]', '', filename)]

        # Add sprite namespace if required
        if self.sprite.config['css_sprite_namespace']:
            sprite_name = re.sub(r'[^\w\-_]', '', self.sprite.name)
            sprite_namespace = self.sprite.config['css_sprite_namespace']

            # Support legacy 0.4 format
            sprite_namespace = sprite_namespace.replace("%(sprite)s", "{sprite_name}")
            namespace.insert(0, sprite_namespace.format(sprite_name=sprite_name))

        # Add global namespace if required
        if self.sprite.config['css_namespace']:
            namespace.insert(0, self.sprite.config['css_namespace'])

        # Handle CamelCase separator
        if self.sprite.config['css_separator'] == self.camelcase_separator:
            namespace = [n[:1].title() + n[1:] if i > 0 else n for i, n in enumerate(namespace)]
            separator = ''

        label = separator.join(namespace)
        pseudo = ''

        css_pseudo_class_separator = self.sprite.config['css_pseudo_class_separator']
        if css_pseudo_class_separator in filename:
            pseudo_classes = [p for p in filename.split(css_pseudo_class_separator) if p in self.css_pseudo_classes]

            # If present add this pseudo class as pseudo an remove it from the label
            if pseudo_classes:
                for p in pseudo_classes:
                    label = label.replace('{0}{1}'.format(css_pseudo_class_separator, p), "")
                pseudo = ''.join(map(lambda x: ':{0}'.format(x), pseudo_classes))

        return label, pseudo
# END Formats/CSS

# START Formats/SCSS
class ScssFormat(CssFormat):

    extension = 'scss'

    @classmethod
    def populate_argument_parser(cls, parser):
        parser.add_argument("--scss", dest="scss_dir", nargs='?', const=True, default=False, metavar='DIR', help="Generate SCSS files and optionally where")
        parser.add_argument("--scss-template", dest="scss_template", default=None, metavar='DIR', help="Template to use to generate the SCSS output.")
                           
# END Formats/SCSS

# START Lists
formats = {'css': CssFormat,
           'img': ImageFormat,
           'scss': ScssFormat}

algorithms = {'square': SquareAlgorithm}
# END Lists

# START Exceptions
class GlueError(Exception):
    """Base Exception class for glue Errors."""
    error_code = 999


class PILUnavailableError(GlueError):
    """Raised if some PIL decoder isn't available."""
    error_code = 2


class ValidationError(GlueError):
    """Raised by formats or sprites while ."""
    error_code = 3


class SourceImagesNotFoundError(GlueError):
    """Raised if a folder doesn't contain any valid image."""
    error_code = 4


class NoSpritesFoldersFoundError(GlueError):
    """Raised if no sprites folders could be found."""
    error_code = 5
# END Exceptions

# START Main
def main(argv=None):

    argv = (argv or sys.argv)[1:]

    parser = argparse.ArgumentParser(description="%(prog)s [source | --source | -s] [output | --output | -o]")

    parser.add_argument("--source", "-s", dest="source", type=str, default=None, help="Source path")
    parser.add_argument("--output", "-o", dest="output", type=str, default=None, help="Output path")
    parser.add_argument("-q", "--quiet", dest="quiet", action='store_true', default=False, help="Suppress all normal output")
    parser.add_argument("-r", "--recursive", dest="recursive", action='store_true', default=False, help="Read directories recursively and add all the images to the same sprite.")
    parser.add_argument("--follow-links", dest="follow_links", action='store_true', default=False, help="Follow symbolic links.")
    parser.add_argument("-f", "--force", dest="force", action='store_true', default=False, help="Force glue to create every sprite image and metadata file even if they already exist in the output directory.")
    parser.add_argument("-v", "--version", action="version", version='%(prog)s ' + __version__, help="Show program's version number and exit")
    
    parser.add_argument("--algorithm", "-a", dest="algorithm", metavar='NAME', type=str, default='square', help="Allocation algorithm: square (default)")
    parser.add_argument("--ordering", dest="algorithm_ordering", metavar='NAME', type=str, default='maxside', help="Ordering criteria: maxside (default)")

    # Populate the parser with options required by other formats
    for format in formats.values():
        format.populate_argument_parser(parser)


    # Parse input
    options, args = parser.parse_known_args(argv)


    # Get the list of enabled formats
    options.enabled_formats = [f for f in formats if getattr(options, '{0}_dir'.format(f), False)]

    # If there is only one enabled format (img) or if there are two (img, html)
    # this means glue is been executed without any specific main format.
    # In order to keep the legacy API we need to enable css.
    # As consequence there is no way to make glue only generate the sprite
    # image and the html file without generating the css file too.
    if set(options.enabled_formats) in (set(['img']), set(['img', 'html'])) and options.generate_css:
        options.enabled_formats.append('css')
        setattr(options, "css_dir", True)

    if not options.generate_image:
        options.enabled_formats.remove('img')


    extra = 0
    # Get the source from the source option or the first positional argument
    if not options.source and args:
        options.source = args[0]
        extra += 1

    # Get the output from the output option or the second positional argument
    if not options.output and args[extra:]:
        options.output = args[extra]

    # Check if source is available
    if options.source is None:
        parser.error(("You must provide the folder containing the sprites "
                      "using the first positional argument or --source."))

    # Make absolute both source and output if present
    if not os.path.isdir(options.source):
        parser.error("Directory not found: '{0}'".format(options.source))

    options.source = os.path.abspath(options.source)
    if options.output:
        options.output = os.path.abspath(options.output)

    # Check that both the source and the output are present. Output "enough"
    # information can be tricky as you can choose different outputs for each
    # of the available formats. If it is present make it absolute.
    if not options.source:
        parser.error(("Source required. Please specify a source using "
                      "--source or the first positional argument."))

    if options.output:
        for format in options.enabled_formats:
            format_option = '{0}_dir'.format(format)
            path = getattr(options, format_option)

            if isinstance(path, bool) and path:
                setattr(options, format_option, options.output)
    else:
        if options.generate_image and not options.img_dir:
            parser.error(("Output required. Please specify an output for "
                          "the sprite image using --output, the second "
                          "positional argument or --img=<DIR>"))

        for format in options.enabled_formats:
            format_option = '{0}_dir'.format(format)
            path = getattr(options, format_option)

            if isinstance(path, bool) or not path:
                parser.error(("{0} output required. Please specify an output "
                              "for {0} using --output, the second "
                              "positional argument or --{0}=<DIR>".format(format)))
            else:
                setattr(options, format_option, os.path.abspath(path))

    # If the img format is not enabled, we still need to know where the sprites
    # were generated. As img is not an enabled format img_dir would be empty
    # if --img was not userd. If this is the case we need to use whatever is
    # the output value.
    if not options.generate_image and isinstance(options.img_dir, bool):
        options.img_dir = options.output

    # Apply formats constraints
    for format in options.enabled_formats:
        formats[format].apply_parser_contraints(parser, options)

    manager_cls = SimpleManager

    manager = manager_cls(**vars(options))

    try:
        if options.quiet:
            with redirect_stdout():
                manager.process()
        else:
            manager.process()
    except ValidationError as e:
        sys.stderr.write(e.args[0])
        return e.error_code
    except SourceImagesNotFoundError as e:
        sys.stderr.write(f"Error: No images found in {e.args[0]}.\n")
        return e.error_code
    except NoSpritesFoldersFoundError as e:
        sys.stderr.write(f"Error: No sprites folders found in {e.args[0]}.\n")
        return e.error_code
    except PILUnavailableError as e:
        sys.stderr.write(f"Error: PIL {e.args[0]} decoder is unavailable. Please read the documentation and install it before spriting this kind of images.\n")
        return e.error_code
    except Exception:
        import platform
        import traceback
        sys.stderr.write("\n")
        sys.stderr.write("=" * 80)
        sys.stderr.write("\nYou've found a bug! Please, raise an issue attaching the following traceback\n")
        sys.stderr.write("https://github.com/jorgebastida/glue/issues/new\n")
        sys.stderr.write("-" * 80)
        sys.stderr.write("\n")
        sys.stderr.write("Version: {0}\n".format(__version__))
        sys.stderr.write("Python: {0}\n".format(sys.version))
        sys.stderr.write("PIL version: {0}\n".format(PILVersion))
        sys.stderr.write("Platform: {0}\n".format(platform.platform()))
        sys.stderr.write("Config: {0}\n".format(vars(options)))
        sys.stderr.write("Args: {0}\n\n".format(sys.argv))
        sys.stderr.write(traceback.format_exc())
        sys.stderr.write("=" * 80)
        sys.stderr.write("\n")
        import pdb
        pdb.post_mortem(sys.exc_info()[-1])
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
# END Main
